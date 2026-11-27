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
        if self.kind == "select":
            return f"select__{self.config}"
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
        "reason": "cache 4-30 Hz con inf in float16 (campioni saturati filtrati): loss NaN"} for s in range(5)},    # CBraMod con EA / z-score: riscalatura all'RMS dei primi 5 soggetti, gonfiato dagli artefatti di sub003/004
    **{jid: {"cutoff": 1795534842,
             "reason": "riscalatura dopo EA/z-score dominata dagli artefatti: training bloccato al caso"}
       for jid in ['cbramod_lora_r2_ea__split_fm__f0__s0__emo9__base', 'cbramod_lora_r2_ea__split_fm__f0__s1__emo9__base', 'cbramod_lora_r2_ea__split_fm__f0__s2__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f0__s0__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f0__s1__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f0__s2__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f1__s0__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f1__s1__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f1__s2__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f2__s0__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f2__s1__emo9__base', 'cbramod_lora_r2_ea__split_fm_lco__f2__s2__emo9__base', 'cbramod_lora_r2_zsub__split_fm__f0__s0__emo9__base', 'cbramod_lora_r2_zsub__split_fm__f0__s1__emo9__base', 'cbramod_lora_r2_zsub__split_fm__f0__s2__emo9__base']},
}


# Notebook F (2026-11-26): la configurazione scelta da Optuna (trial 27, un solo seed) collassa a una classe
# costante in 4/9 job a due fasi pre-addestrati sul leave-clip-out, 1/3 su split FM, 2/6 permutati e 12/12 da
# zero (BA di validation = 1/9 dalla fase LP in poi). Si ripetono tutti con la configurazione riselezionata
# su 3 seed (coda F, job `select__cbramod_twophase`).
INVALIDATED.update({
    jid: {"cutoff": 1795780015, "reason": "ricetta a due fasi del trial 27: collasso a classe costante; ripetuto "
                                         "con la configurazione riselezionata su 3 seed"}
    for jid in [f"cbramod_twophase_{v}__{p}__f{f}__s{s}__emo9__base" for v in ("pre", "scratch")
                for p, folds in (("split_fm", [0]), ("split_fm_lco", [0, 1, 2])) for f in folds for s in (3407, 1, 2)]
    + [f"cbramod_twophase_pre__split_fm_lco__f{f}__s3407__emo9__perm{k}" for f in range(3) for k in range(2)]})


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
    # (2026-11-24) dopo il notebook C: (a) ripetizione di CBraMod con EA / z-score con la riscalatura robusta;
    # (b) scomposizione della caduta a 4-30 Hz: solo passa-alto 4 Hz e solo passa-basso 30 Hz (3 seed)
    J += train_jobs("cbramod_lora_r2_ea", ["split_fm", "split_fm_lco"], SEEDS3)
    J += train_jobs("cbramod_lora_r2_zsub", ["split_fm"], SEEDS3)
    J += [Job("train", "cbramod_lora_r2", "split_fm", 0, s, band=b) for b in ([4, 47], [0, 30]) for s in SEEDS3]
    J += [Job("train", "tsception_ea", "split_fm", 0, s, band=b) for b in ([4, 47], [0, 30]) for s in SEEDS3]
    # (c) bordo frontale/oculare contro temporale (CBraMod perde 4.5 punti senza i canali di bordo)
    J += [Job("train", "cbramod_lora_r2", "split_fm", 0, s, drop=d) for d in ("ocular4", "temporal4") for s in SEEDS3]
    J += [Job("train", "cbramod_lora_r2", "split_fm", 0, SEEDS3[k], drop=f"rand4_d{k}") for k in range(3)]
    # (2026-11-25) esito: il crollo viene tutto da sotto 4 Hz (passa-alto 4 Hz -38 punti, passa-basso 30 Hz +1).
    # (d) leave-clip-out con passa-alto 4 Hz: tolto il contenuto lento, sulle clip nuove cambia qualcosa?
    J += train_jobs("cbramod_lora_r2", ["split_fm_lco"], SEEDS3, band=[4, 47])
    # (e) dove sta l'informazione sotto 4 Hz: solo 0-4 Hz; passa-alto 1 Hz e 2 Hz
    J += [Job("train", "cbramod_lora_r2", "split_fm", 0, s, band=b) for b in ([0, 4], [1, 47], [2, 47])
          for s in SEEDS3]
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


def queue_nbD() -> list[Job]:
    """10 fold sui soggetti (tutti i 123 soggetti finiscono in test una volta), modelli compatti e feature.
    Motivo: potenza statistica (23 -> 123 soggetti appaiati) e indipendenza da un singolo split.
    Ordine di priorità: (1) lnso10 per il gate di potenza, (2) leave-clip-out, (3) controllo della quantità
    di dati + permutazioni per le correzioni, (4) feature, (5) EMG (facoltativo, in coda).
    CBraMod sui 10 fold: coda separata, da definire dopo il notebook C (config finale CBraMod)."""
    J = []
    # (1) gate di potenza: TSception vs DE (e vs caso) su 123 soggetti
    J += train_jobs("const_majority", ["lnso10"], [0])
    J += train_jobs("tsception_ea", ["lnso10"], SEEDS3)
    J += train_jobs("de_asym_lr", ["lnso10"], [0])
    J += train_jobs("de_asym_lr_zsub", ["lnso10"], [0])
    # (2) leave-clip-out: Δstim appaiato e generalizzazione a stimoli nuovi con piena potenza
    J += train_jobs("tsception_ea", ["lnso10_lco"], SEEDS3)
    # (3) quantità di dati vs novità dello stimolo (H3) + permutazioni a livello di clip per le correzioni
    J += sizematched_jobs("tsception_ea", [0], draws=[0], n_perm_models=1, protocol="lnso10_sizematched")
    J += perm_jobs("tsception_ea", "lnso10_lco", perms=[0])
    # (3b) aggiunta 2026-11-26: CBraMod LoRA sul leave-clip-out dei 10 fold. La prima sessione (codice vecchio)
    # ne ha completati 15 su 30 (soggetti 0-59: clip nuove 0.122, +1.1 sul caso, p 0.04): si completano gli
    # altri 15 per avere CBraMod sulle clip nuove con tutti i 123 soggetti (i 15 fatti vengono saltati)
    J += train_jobs("cbramod_lora_r2", ["lnso10_lco"], [0])
    # (4) feature (CPU): leave-clip-out e controllo della quantità di dati con permutazioni interne
    J += [Job("train", c, "lnso10_lco", f, 0, n_perm=20) for c in ("de_asym_lr_zsub",)
          for f in _folds("lnso10_lco")]
    J += [Job("train", "de_asym_lr_zsub", "lnso10_sizematched", f, 0, draw=0, n_perm=20)
          for f in _folds("lnso10_sizematched")]
    # (5) facoltativo: EMG con piena potenza (1 seed, 2 estrazioni di controllo)
    J += [Job("train", "tsception_ea", "lnso10", f, 0, band=[4, 30], drop=d) for d in ("edge8", "rand8_d0", "rand8_d1")
          for f in _folds("lnso10")]
    return J


def queue_nbE() -> list[Job]:
    """EEGNet con la metodologia originale nel protocollo del benchmark CBraMod (account 3).
    Domande: (1) il divario 0.229 (nostro EEGNet, 4 s + EA + nostra ricetta) contro 0.409 (EEGNet riportato da
    CBraMod) viene da finestra, ricetta o allineamento? (2) anche EEGNet ben addestrato è memorizzazione della
    clip (leave-clip-out)? Ordine: split FM (3 ricette x 3 seed) -> leave-clip-out -> permutati."""
    J = []
    J += train_jobs("eegnet_lawhern", ["split_fm"], CBRAMOD_SEEDS3)
    J += train_jobs("eegnet_cbramod_recipe", ["split_fm"], CBRAMOD_SEEDS3)
    J += train_jobs("eegnet_lawhern_ea", ["split_fm"], CBRAMOD_SEEDS3)
    J += train_jobs("eegnet_lawhern", ["split_fm_lco"], CBRAMOD_SEEDS3)
    J += train_jobs("eegnet_cbramod_recipe", ["split_fm_lco"], CBRAMOD_SEEDS3)
    J += perm_jobs("eegnet_lawhern", "split_fm_lco", perms=range(2), seed=3407)
    # Aggiunte dopo la prima sessione (2026-11-24): EEGNet originale + EA arriva a 0.455 su split FM.
    # (a) regge su clip nuove? leave-clip-out + 2 modelli permutati per fold (indice appaiato corretto)
    J += train_jobs("eegnet_lawhern_ea", ["split_fm_lco"], CBRAMOD_SEEDS3)
    J += perm_jobs("eegnet_lawhern_ea", "split_fm_lco", perms=range(2), seed=3407)
    # (b) TSception con la metodologia originale e finestre da 10 s, con e senza EA (stesso problema di finestra)
    J += train_jobs("tsception_ding", ["split_fm"], CBRAMOD_SEEDS3)
    J += train_jobs("tsception_ding_ea", ["split_fm"], CBRAMOD_SEEDS3)
    # (c) scomposizione dati / stimolo a parità di quantità di dati (H3 per il miglior modello compatto)
    J += sizematched_jobs("eegnet_lawhern_ea", CBRAMOD_SEEDS3, draws=[0], n_perm_models=2)
    # (d) TSception originale + EA su clip nuove (+ 2 permutati per fold)
    J += train_jobs("tsception_ding_ea", ["split_fm_lco"], CBRAMOD_SEEDS3)
    J += perm_jobs("tsception_ding_ea", "split_fm_lco", perms=range(2), seed=3407)
    # (e) da dove viene il salto da 0.229 a 0.455: finestra da 4 s con la stessa ricetta
    J += train_jobs("eegnet_lawhern_ea_w4", ["split_fm"], CBRAMOD_SEEDS3)
    # (f) training più lungo (la validation migliorava ancora a 200 epoche)
    J += train_jobs("eegnet_lawhern_ea_long", ["split_fm"], CBRAMOD_SEEDS3)
    # (2026-11-25) per CBraMod l'informazione sta sotto 4 Hz; finestre da 10 s valgono +19 punti per EEGNet.
    # (g) EEGNet originale + EA usa lo stesso contenuto lento? passa-alto 4 Hz e solo 0-4 Hz (3 seed)
    J += [Job("train", "eegnet_lawhern_ea", "split_fm", 0, s, band=b) for b in ([4, 47], [0, 4])
          for s in CBRAMOD_SEEDS3]
    # (h) leave-clip-out con passa-alto 4 Hz
    J += train_jobs("eegnet_lawhern_ea", ["split_fm_lco"], CBRAMOD_SEEDS3, band=[4, 47])
    # (2026-11-25) esito: a 4-47 Hz le clip nuove salgono a 0.136 (+2.5 sul caso). (i) 2 modelli permutati per fold
    # a 4-47 Hz: il divario viste/non viste che resta (18.6 punti) è ancora memorizzazione?
    J += [Job("train", "eegnet_lawhern_ea", "split_fm_lco", f, 3407, perm=k, band=[4, 47])
          for k in range(2) for f in _folds("split_fm_lco")]
    return J


# ------------------------------------------------------------------ Sessione 02
def queue_F() -> list[Job]:
    """Notebook F (ricetta a due fasi e pretraining), in ordine di priorità:
    Optuna cbramod_twophase (30 trial, 2 shard) -> controllo di sanità (split_fm, 1 seed: la BA di VALIDATION
    deve superare 0.20, altrimenti il job fallisce e i job della Parte A non partono) -> Parte A
    (pre-addestrato e da zero su split_fm e leave-clip-out, 2 permutati per fold) -> Parte B (da zero con
    warm-up e pre-addestrato con warm-up su leave-clip-out)."""
    from ..hpo.search_spaces import N_TRIALS
    J = hpo_jobs("cbramod_twophase", N_TRIALS["cbramod_twophase"])
    hpo_ids = [j.job_id for j in J]
    # Riselezione su più seed: i 4 migliori trial di Optuna (tutti con testa appiattita), 3 seed ciascuno su
    # split FM; vince la media della BA di validation. Il trial 27 da solo collassava in metà dei seed.
    top = _hpo_best_full("cbramod_twophase").get("top", [])
    cands = {f"cand{c['trial']}": c for c in top}
    cjobs = [Job("train", "cbramod_twophase_pre", "split_fm", 0, s,
                 overrides={"hpo_params": c["params"], "tag": tag}, depends_on=hpo_ids)
             for tag, c in cands.items() for s in CBRAMOD_SEEDS3]
    J += cjobs
    sel = Job("select", "cbramod_twophase", overrides={"candidates": cands, "seeds": CBRAMOD_SEEDS3},
              depends_on=[j.job_id for j in cjobs] or hpo_ids)
    J.append(sel)
    sanity = Job("train", "cbramod_twophase_pre", "split_fm", 0, 3407, overrides={"sanity_min_val": 0.20},
                 depends_on=[sel.job_id])
    J.append(sanity)
    dep = [sanity.job_id]
    J += train_jobs("cbramod_twophase_pre", ["split_fm"], [1, 2], depends_on=dep)
    J += train_jobs("cbramod_twophase_pre", ["split_fm_lco"], CBRAMOD_SEEDS3, depends_on=dep)
    J += perm_jobs("cbramod_twophase_pre", "split_fm_lco", perms=range(2), seed=3407, deps=dep)
    J += train_jobs("cbramod_twophase_scratch", ["split_fm", "split_fm_lco"], CBRAMOD_SEEDS3, depends_on=dep)
    # Parte B: il pretraining sulle clip non viste (le ricette con warm-up esistono solo su split_fm)
    J += train_jobs("cbramod_scratch_ft_warmup", ["split_fm_lco"], CBRAMOD_SEEDS3)
    J += train_jobs("cbramod_ft_official_warmup", ["split_fm_lco"], CBRAMOD_SEEDS3)
    return J


def queue_F_bis() -> list[Job]:
    """Ablazione A.4 (PREPARATA, da lanciare solo se cbramod_twophase_pre sulle clip non viste supera il caso:
    Wilcoxon unilaterale per soggetto contro 1/9, p < 0.05). Una modifica alla volta rispetto al migliore,
    leave-clip-out, 3 seed. La scelta della prima riga dipende dalla testa scelta da Optuna: se è già
    flat_official si usa la migliore configurazione con testa a pooling, e le due varianti della
    standardizzazione non hanno senso (la testa appiattita non la usa)."""
    best = _hpo_best_params("cbramod_twophase")
    flat_best = best.get("head") == "flat_official"
    cfgs = ["cbramod_twophase_pre_bestpool"] if flat_best else \
        ["cbramod_twophase_pre_flathead", "cbramod_twophase_pre_nonorm", "cbramod_twophase_pre_heldoutnorm"]
    cfgs.append("cbramod_official_poolhead")
    J = []
    for c in cfgs:
        J += train_jobs(c, ["split_fm_lco"], CBRAMOD_SEEDS3)
    return J


def _hpo_best_full(study: str) -> dict:
    p = Path(__file__).resolve().parents[1] / "hpo" / "best" / f"{study}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _hpo_best_params(study: str) -> dict:
    p = Path(__file__).resolve().parents[1] / "hpo" / "best" / f"{study}.json"
    return json.loads(p.read_text()).get("params", {}) if p.exists() else {}


REVE_SEEDS_GATE = [42, 43, 44, 45, 46]
REVE_SEEDS_GATE_R3 = [47, 48, 49, 50, 51]


def queue_G() -> list[Job]:
    """Notebook G (REVE-Base), in ordine di priorità (§C.3): se il tempo finisce, i blocchi 1-4 devono essere
    completi. Stesse definizioni di CBraMod: soggetti, fold di clip, seed, mappe di permutazione (indice della
    permutazione), estrazioni del controllo a parità di dati, insiemi di canali, tagli di banda.
    Stime a priori (est_min) da ricalibrare dopo il primo job: ~45 min split_fm, ~35 min per fold LCO."""
    R, E = "reve_base", {"split_fm": 45, "split_fm_lco": 35, "split_fm_sizematched": 35}

    def tj(config, protocols, seeds, **kw):
        return [Job("train", config, p, f, s, est_min=E.get(p, 45), **kw)
                for p in protocols for f in _folds(p) for s in seeds]
    J = []
    # 1. Gate R: R1 (solo LP, 1 seed) e R2 (LP+FT, 5 seed); la sonda della clip (blocco 8) sul primo seed
    J += tj("reve_base_lp", ["split_fm"], [42])
    J += [Job("train", R, "split_fm", 0, s, probe=(s == 42), est_min=45) for s in REVE_SEEDS_GATE]
    # 2. split_fm con T_k: dai job di R2 (condizioni tk0..2 calcolate nello stesso job)
    # 3. leave-clip-out (risultato centrale)
    J += tj(R, ["split_fm_lco"], SEEDS3)
    # 4. permutati sul leave-clip-out
    J += [Job("train", R, "split_fm_lco", f, 0, perm=k, est_min=35) for k in range(2) for f in _folds("split_fm_lco")]
    # 5. a parità di dati (+ 2 permutati), estrazione 0 come CBraMod
    J += [dataclasses_replace(j, est_min=35) for j in sizematched_jobs(R, SEEDS3, draws=[0], n_perm_models=2)]
    # 6. bande: solo 4-30, solo passa-alto 4 Hz, solo passa-basso 30 Hz (3 seed)
    J += [Job("train", R, "split_fm", 0, s, band=b, est_min=45) for b in ([4, 30], [4, 47], [0, 30]) for s in SEEDS3]
    # 7. canali (stesse estrazioni di CBraMod): bordo, metà frontale, metà temporale, casuali; 4-30 con bordo e casuali
    J += [Job("train", R, "split_fm", 0, s, drop="edge8", est_min=45) for s in SEEDS3]
    J += [Job("train", R, "split_fm", 0, s, drop=d, est_min=45) for d in ("ocular4", "temporal4") for s in SEEDS3]
    J += [Job("train", R, "split_fm", 0, SEEDS3[k], drop=f"rand4_d{k}", est_min=45) for k in range(3)]
    J += [Job("train", R, "split_fm", 0, SEEDS3[k], drop=f"rand8_d{k}", est_min=45) for k in range(3)]
    J += [Job("train", R, "split_fm", 0, s, band=[4, 30], drop="edge8", est_min=45) for s in SEEDS3]
    J += [Job("train", R, "split_fm", 0, SEEDS3[k], band=[4, 30], drop=f"rand8_d{k}", est_min=45) for k in range(3)]
    # 8. sonda dell'identità della clip: nel job R2 con seed 42 (contesto della query cls e media dei token)
    # 9. valenza binaria (positivo contro negativo): REVE e, perché non esiste, CBraMod ricetta ufficiale
    J += tj(R, ["split_fm", "split_fm_lco"], SEEDS3, task="val2")
    J += train_jobs("cbramod_ft_official", ["split_fm", "split_fm_lco"], SEEDS3, task="val2")
    # R3 (se resta tempo): altri 5 seed del Gate
    J += [Job("train", R, "split_fm", 0, s, est_min=45) for s in REVE_SEEDS_GATE_R3]
    # 10-11. opzionali: REVE da zero; LoRA rango 16
    J += tj("reve_base_scratch", ["split_fm", "split_fm_lco"], SEEDS3)
    J += tj("reve_base_lora16", ["split_fm", "split_fm_lco"], SEEDS3)
    return J


def dataclasses_replace(job: Job, **kw) -> Job:
    from dataclasses import replace
    return replace(job, **kw)


QUEUES = {"nb0_smoke": queue_nb0, "nbA_baselines_tsception": queue_nbA, "nbB_cbramod_gate": queue_nbB,
          "nbC_cbramod_main": queue_nbC, "nbD_lnso10": queue_nbD, "nbE_eegnet": queue_nbE,
          "F": queue_F, "F_bis": queue_F_bis, "G": queue_G}


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
