"""Coda di job condivisa su file: (config, protocollo, fold, seed, variante). Salta i job completati.

    python -m pianob.experiments.queue --out queues/     # rigenera le code dei notebook
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import kaggle_env
from ..data.splits import n_folds

DONE_STATUSES = {"completed", "early_stopped"}


@dataclass
class Job:
    kind: str                       # train | hpo | report
    config: str
    protocol: str = "split_fm"
    fold: int = 0
    seed: int = 0
    task: str = "emo9"
    band: list | None = None        # es. [4, 30]
    drop: str | None = None         # edge8 | rand8_d{k} | edge8_mastoids
    perm: int | None = None         # permutazione delle label a livello di clip
    draw: int | None = None         # estrazione del sottocampionamento (protocolli *_sizematched)
    n_trials: int = 0               # hpo
    shard: int = 0                  # hpo: più worker sullo stesso studio
    probe: bool = False             # sonda dell'identità della clip a fine training
    n_perm: int = 0                 # baseline su feature: riaddestramenti permutati interni al job
    overrides: dict = field(default_factory=dict)   # debug/smoke: max_subjects, hparams, ...
    depends_on: list = field(default_factory=list)
    est_min: float = 0.0

    @property
    def variant(self) -> str:
        parts = []
        if self.band:
            parts.append(f"bp{self.band[0]:g}-{self.band[1]:g}")
        if self.drop == "edge8":
            parts.append("noedge")
        elif self.drop == "edge8_mastoids":
            parts.append("noedgemast")
        elif self.drop:
            parts.append(self.drop.replace("_", ""))
        if self.draw is not None:
            parts.append(f"draw{self.draw}")
        if self.perm is not None:
            parts.append(f"perm{self.perm}")
        if self.overrides.get("tag"):
            parts.append(self.overrides["tag"])
        return "_".join(parts) or "base"

    @property
    def job_id(self) -> str:
        if self.kind == "hpo":
            return f"hpo__{self.config}__shard{self.shard}"
        if self.kind == "report":
            return f"report__{self.config}"
        return f"{self.config}__{self.protocol}__f{self.fold}__s{self.seed}__{self.task}__{self.variant}"

    def to_dict(self):
        d = asdict(self)
        d["job_id"] = self.job_id
        return d

    @staticmethod
    def from_dict(d):
        d = {k: v for k, v in d.items() if k != "job_id"}
        d.setdefault("draw", None)
        return Job(**d)


def save_queue(jobs: list[Job], path: Path) -> None:
    ids = [j.job_id for j in jobs]
    dup = {i for i in ids if ids.count(i) > 1}
    assert not dup, f"job duplicati: {dup}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([j.to_dict() for j in jobs], indent=1))


def load_queue(path: Path) -> list[Job]:
    return [Job.from_dict(d) for d in json.loads(Path(path).read_text())]


# ------------------------------------------------------------------ stato dei job
# Job completati con risultati NON validi: i record precedenti a `cutoff` si ignorano (il job viene
# rieseguito) e le relative righe si scartano nelle analisi.
INVALIDATED = {
    **{f"tsception_ea__split_fm__f0__s{s}__emo9__bp4-30": {
        "cutoff": 1795348800,   # 2026-11-22 12:00 UTC
        "reason": "cache 4-30 Hz con inf in float16 (campioni saturati filtrati): loss NaN"} for s in range(5)},
}


def is_invalid_record(job_id: str, t) -> bool:
    inv = INVALIDATED.get(job_id)
    return inv is not None and (t is None or float(t) < inv["cutoff"])


def invalid_row_mask(df):
    """Righe di risultati prodotte da job invalidati (prima del cutoff), incluse le righe BN-α."""
    import numpy as np
    m = np.zeros(len(df), dtype=bool)
    if df.empty:
        return m
    t = df["t"] if "t" in df.columns else None
    for jid, inv in INVALIDATED.items():
        config, protocol, fold, seed, task, variant = jid.split("__")
        sel = (df.config.astype(str).str.startswith(config) & (df.protocol == protocol)
               & (df.fold == int(fold[1:])) & (df.seed == int(seed[1:])) & (df.task == task)
               & ((df.variant == variant) | df.variant.astype(str).str.startswith(variant + "_bnalpha")))
        old = sel if t is None else sel & (t.isna() | (t < inv["cutoff"]))
        m |= old.values
    return m


def done_status() -> dict[str, dict]:
    out = {}
    for f in (kaggle_env.state_dir() / "done").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if is_invalid_record(r["job_id"], r.get("t")):
                    continue
                prev = out.get(r["job_id"])
                if prev is None or prev["status"] not in DONE_STATUSES:
                    out[r["job_id"]] = r
    return out


def mark(job: Job, status: str, worker: str, minutes: float, extra: dict | None = None) -> None:
    rec = {"job_id": job.job_id, "status": status, "worker": worker, "minutes": round(minutes, 2),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "t": time.time(), **(extra or {})}
    with open(kaggle_env.state_dir() / "done" / f"done_{worker}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def try_claim(job: Job, worker: str) -> bool:
    p = kaggle_env.state_dir() / "locks" / f"{job.job_id}.lock"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, worker.encode())
    os.close(fd)
    return True


def release(job: Job) -> None:
    (kaggle_env.state_dir() / "locks" / f"{job.job_id}.lock").unlink(missing_ok=True)


def is_locked(job: Job) -> bool:
    return (kaggle_env.state_dir() / "locks" / f"{job.job_id}.lock").exists()


# ------------------------------------------------------------------ costruzione delle code
def _folds(protocol):
    return range(n_folds(protocol))


def train_jobs(config, protocols, seeds, task="emo9", draws=(None,), **kw) -> list[Job]:
    return [Job("train", config, p, f, s, task, draw=d, **kw)
            for p in protocols for d in draws for f in _folds(p) for s in seeds]


def sizematched_jobs(config, seeds, draws, n_perm_models=0, protocol="split_fm_sizematched", deps=(),
                     **kw) -> list[Job]:
    """Controllo della quantità di dati: seed x estrazioni x 3 fold di clip, più modelli con label
    permutate a livello di clip (estrazione 0, seed 0) per la correzione di Δstim^size."""
    J = train_jobs(config, [protocol], seeds, draws=draws, depends_on=list(deps), **kw)
    J += [Job("train", config, protocol, f, 0, draw=draws[0], perm=k, depends_on=list(deps), **kw)
          for k in range(n_perm_models) for f in _folds(protocol)]
    return J


def hpo_jobs(study, n_trials, n_shards=2) -> list[Job]:
    return [Job("hpo", study, n_trials=n_trials, shard=k) for k in range(n_shards)]


def emg_jobs(config, seeds, n_draws=5, protocol="split_fm", one_seed_per_draw=False, deps=()) -> list[Job]:
    """Ablazione EMG: 4-30 Hz, senza canali di bordo, entrambe; controllo con 8 canali non di bordo
    casuali (n_draws estrazioni), a banda piena e a 4-30 Hz. La condizione 'piena' è il job base."""
    d_ = list(deps)
    jobs = []
    for s in seeds:
        jobs += [Job("train", config, protocol, 0, s, band=[4, 30], depends_on=d_),
                 Job("train", config, protocol, 0, s, drop="edge8", depends_on=d_),
                 Job("train", config, protocol, 0, s, band=[4, 30], drop="edge8", depends_on=d_)]
    for d in range(n_draws):
        for s in ([seeds[d % len(seeds)]] if one_seed_per_draw else seeds):
            jobs += [Job("train", config, protocol, 0, s, drop=f"rand8_d{d}", depends_on=d_),
                     Job("train", config, protocol, 0, s, band=[4, 30], drop=f"rand8_d{d}", depends_on=d_)]
    return jobs


def perm_jobs(config, protocol, perms, seed=0, deps=(), task="emo9") -> list[Job]:
    return [Job("train", config, protocol, f, seed, task, perm=k, depends_on=list(deps))
            for k in perms for f in _folds(protocol)]


def queue_nb0() -> list[Job]:
    smoke = {"max_subjects": 3, "hparams": {"epochs": 1, "patience": 0}}
    jobs = [Job("train", "const_majority", overrides=dict(smoke)),
            Job("train", "de_asym_lr", "split_fm_lco", overrides={"max_subjects": 3}),
            Job("train", "logpower5_lr", overrides={"max_subjects": 3})]
    for c in ["eegnet_ea", "tsception_ea", "cbramod_ft_official", "cbramod_lora_r2", "cbramod_lora_r2_ea"]:
        jobs.append(Job("train", c, "split_fm_lco" if c == "tsception_ea" else "split_fm",
                        overrides=dict(smoke), probe=(c == "cbramod_lora_r2")))
    return jobs


SEEDS5 = [0, 1, 2, 3, 4]
SEEDS3 = [0, 1, 2]
CBRAMOD_SEEDS3 = [3407, 1, 2]   # 3407 = seed ufficiale


def queue_nbA() -> list[Job]:
    """Account 1: baseline su feature -> Optuna TSception/EEGNet -> TSception/EEGNet -> BN-α
    (incluso nei job tsception_ea) -> ablazione EMG -> permutazioni Δstim."""
    J = []
    J += [Job("train", "const_majority", p, f) for p in ("split_fm", "split_fm_lco") for f in _folds(p)]
    for c in ("logpower5_lr", "de_asym_lr", "de_asym_lr_zsub"):
        J += [Job("train", c, "split_fm", 0, 0, n_perm=100)]
        J += [Job("train", c, "split_fm_lco", f, 0, n_perm=100) for f in _folds("split_fm_lco")]
    hpo_t = hpo_jobs("tsception_ea", 60)
    hpo_e = hpo_jobs("eegnet_ea", 40)
    J += hpo_t + hpo_e
    dt = [j.job_id for j in hpo_t]
    de = [j.job_id for j in hpo_e]
    J += train_jobs("tsception_ea", ["split_fm", "split_fm_lco"], SEEDS5, depends_on=dt)
    J += train_jobs("eegnet_ea", ["split_fm", "split_fm_lco"], SEEDS5, depends_on=de)
    J += train_jobs("tsception_noalign", ["split_fm"], SEEDS5, depends_on=dt)
    J += train_jobs("eegnet_noalign", ["split_fm"], SEEDS5, depends_on=de)
    J += emg_jobs("tsception_ea", SEEDS5, n_draws=5, deps=dt)
    # controllo della quantità di dati per il leave-clip-out (dopo l'ablazione EMG)
    J += sizematched_jobs("tsception_ea", SEEDS5, draws=[0, 1], n_perm_models=5, deps=dt)
    J += [Job("train", "de_asym_lr_zsub", "split_fm_sizematched", f, 0, draw=d, n_perm=50)
          for d in (0, 1) for f in _folds("split_fm_sizematched")]
    # BA_full su T_k per la baseline su feature (il job split_fm originale non aveva le condizioni tk)
    J += [Job("train", "de_asym_lr_zsub", "split_fm", 0, 0, overrides={"tag": "tkeval"})]
    J += perm_jobs("tsception_ea", "split_fm_lco", perms=range(5), deps=dt)
    J += [Job("train", "tsception_ea", "split_fm", 0, 0, probe=True, overrides={"tag": "probe"}, depends_on=dt)]
    J += train_jobs("tsception_ea", ["split_fm"], SEEDS3, task="val2", depends_on=dt)
    J += [Job("train", c, p, f, 0, task="val2") for c in ("const_majority", "de_asym_lr")
          for p in ("split_fm",) for f in _folds(p)]
    # Valenza binaria su stimoli nuovi (aggiunta 2026-11-23): leave-clip-out + controllo della quantità di
    # dati, con permutazioni per entrambe le correzioni. BA_full su T_k: TSception dalle predizioni salvate
    # dei job val2 su split_fm; DE con un job di rivalutazione 'tkeval'.
    J += train_jobs("tsception_ea", ["split_fm_lco"], SEEDS3, task="val2", depends_on=dt)
    J += perm_jobs("tsception_ea", "split_fm_lco", perms=range(3), deps=dt, task="val2")
    J += sizematched_jobs("tsception_ea", SEEDS3, draws=[0, 1], n_perm_models=3, deps=dt, task="val2")
    J += [Job("train", "const_majority", "split_fm_lco", f, 0, task="val2") for f in _folds("split_fm_lco")]
    J += [Job("train", "de_asym_lr", "split_fm_lco", f, 0, task="val2", n_perm=100) for f in _folds("split_fm_lco")]
    J += [Job("train", "de_asym_lr", "split_fm_sizematched", f, 0, task="val2", draw=d, n_perm=50)
          for d in (0, 1) for f in _folds("split_fm_sizematched")]
    J += [Job("train", "de_asym_lr", "split_fm", 0, 0, task="val2", overrides={"tag": "tkeval"})]
    return J


def queue_nbB() -> list[Job]:
    """Account 2: Gate 1 (ricetta ufficiale) -> init casuale -> Optuna LoRA."""
    J = train_jobs("cbramod_ft_official", ["split_fm"], CBRAMOD_SEEDS3)
    J += train_jobs("cbramod_scratch_ft", ["split_fm"], CBRAMOD_SEEDS3)
    J += hpo_jobs("cbramod_lora_r2", 30)
    # Diagnostica dell'init casuale (aggiunta 2026-11-24): 1 seed per variante, poi 3 seed per quella che impara
    J += [Job("train", c, "split_fm", 0, 3407) for c in
          ("cbramod_scratch_ft_definit", "cbramod_scratch_ft_warmup", "cbramod_scratch_ft_definit_warmup")]
    # Esito (2026-11-24): impara solo con il warm-up delle LR -> 3 seed del da-zero con warm-up e, per H1 a
    # ricetta identica, il pre-addestrato con lo stesso warm-up
    J += [Job("train", "cbramod_scratch_ft_warmup", "split_fm", 0, s) for s in (1, 2)]
    J += train_jobs("cbramod_ft_official_warmup", ["split_fm"], CBRAMOD_SEEDS3)
    return J


def queue_nbC() -> list[Job]:
    """Account 3 (dopo il Gate 1 e con l'Optuna LoRA disponibile)."""
    J = train_jobs("cbramod_lora_r2", ["split_fm"], SEEDS3, probe=False)
    J += train_jobs("cbramod_lora_r2", ["split_fm_lco"], SEEDS5)
    J += train_jobs("cbramod_ft_official", ["split_fm_lco"], SEEDS5)
    J += train_jobs("cbramod_lora_r2_ea", ["split_fm", "split_fm_lco"], SEEDS3)
    J += sizematched_jobs("cbramod_lora_r2", SEEDS3, draws=[0], n_perm_models=2)   # prima dell'ablazione EMG
    J += emg_jobs("cbramod_lora_r2", SEEDS3, n_draws=3, one_seed_per_draw=True)
    J += [Job("train", "cbramod_lora_r2", "split_fm", 0, 0, probe=True, overrides={"tag": "probe"})]
    J += perm_jobs("cbramod_lora_r2", "split_fm_lco", perms=range(3))
    J += train_jobs("cbramod_lora_r2_zsub", ["split_fm"], SEEDS3)
    return J


def queue_nbD(final_configs=("tsception_ea", "cbramod_lora_r2", "de_asym_lr")) -> list[Job]:
    """lnso10 (3 seed) + lnso10_lco per le config finali. Per CBraMod lnso10_lco con 1 seed (30 job da
    ~20 min su T4): il budget completo a 3 seed supererebbe ~50 GPU-h. Da rigenerare quando le config
    finali sono scelte (Gate di potenza dopo le prime due config)."""
    J = []
    for c in final_configs:
        if c.startswith(("de_", "logpower", "const")):
            J += train_jobs(c, ["lnso10", "lnso10_lco"], [0])
        elif c.startswith("cbramod"):
            J += train_jobs(c, ["lnso10"], SEEDS3) + train_jobs(c, ["lnso10_lco"], [0])
        else:
            J += train_jobs(c, ["lnso10", "lnso10_lco"], SEEDS3)
    # controllo della quantità di dati: solo per le 2 config finali, 1 seed x 1 estrazione
    for c in final_configs[:2]:
        J += train_jobs(c, ["lnso10_sizematched"], [0], draws=[0])
    return J


QUEUES = {"nb0_smoke": queue_nb0, "nbA_baselines_tsception": queue_nbA, "nbB_cbramod_gate": queue_nbB,
          "nbC_cbramod_main": queue_nbC, "nbD_lnso10": queue_nbD}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="queues")
    a = ap.parse_args()
    for name, fn in QUEUES.items():
        jobs = fn()
        save_queue(jobs, Path(a.out) / f"{name}.json")
        print(f"{name}: {len(jobs)} job")


if __name__ == "__main__":
    main()
