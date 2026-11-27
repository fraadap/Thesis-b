"""Riepilogo compatto da copiare in chat (tabella markdown + JSON) e zip dei risultati."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import kaggle_env

GATE1_TARGET = 0.5509
GATE1_TOL = 0.02


def load_rows(paths=None) -> pd.DataFrame:
    paths = paths or sorted((kaggle_env.state_dir() / "results").glob("rows_*.jsonl"))
    recs = []
    for p in paths:
        recs += kaggle_env.read_jsonl(p)
    recs += _tk_rows_from_state(recs)
    df = pd.DataFrame(recs)
    if not df.empty:
        from .experiments.queue import invalid_row_mask
        df = df[~invalid_row_mask(df)]
        if "t" in df.columns:
            df = df.sort_values("t", na_position="first", kind="stable")
        df = df.drop_duplicates(subset=["config", "protocol", "fold", "seed", "variant", "test_condition",
                                        "task", "subject"], keep="last")
    return df


def _tk_rows_from_state(recs) -> list[dict]:
    """BA_full su T_k per i job split_fm/lnso10 completati prima che esistesse la condizione 'tk{k}'."""
    from .eval.shortcuts import tk_rows_from_preds
    have = {(r["config"], r["protocol"], r["fold"], r["seed"], r["task"], r["variant"])
            for r in recs if str(r.get("test_condition", "")).startswith("tk")}
    out = []
    for f in sorted((kaggle_env.state_dir() / "artifacts" / "preds").glob("*.npz")):
        p = f.stem.split("__")
        if len(p) == 6 and (p[0], p[1], int(p[2][1:]), int(p[3][1:]), p[4], p[5]) not in have:
            with np.load(f) as z:
                out += tk_rows_from_preds(f.stem, z)
    return out


def attach_job_times(jobs: list[dict], done: list[dict]) -> list[dict]:
    """Le righe jobs_*.jsonl del codice più vecchio non hanno "t": lo prende dalla n-esima riga done dello
    stesso job nello stesso worker (i due file sono scritti nello stesso ordine)."""
    from collections import defaultdict
    times = defaultdict(list)
    from .experiments.queue import DONE_STATUSES
    for d in done:   # solo gli esiti che scrivono un record (non failed/interrupted)
        if d.get("status") in DONE_STATUSES:
            times[d.get("job_id")].append(d.get("t"))
    seen = defaultdict(int)
    for j in jobs:
        i = seen[j["job_id"]]
        seen[j["job_id"]] += 1
        if j.get("t") is None and i < len(times[j["job_id"]]):
            j["t"] = times[j["job_id"]][i]
    return jobs


def latest_valid_jobs(jobs: list[dict]) -> list[dict]:
    """Un record per job: scarta quelli invalidati e, a parità di job, tiene il più recente."""
    from .experiments.queue import is_invalid_record
    ok = [j for j in jobs if not is_invalid_record(j["job_id"], j.get("t"))]
    ok.sort(key=lambda j: (j.get("t") is not None, j.get("t") or 0))
    return list({j["job_id"]: j for j in ok}.values())


def _read_jsonl(p) -> list[dict]:
    return kaggle_env.read_jsonl(p)


def load_jobs(paths=None) -> list[dict]:
    paths = paths or sorted((kaggle_env.state_dir() / "results").glob("jobs_*.jsonl"))
    out = []
    for p in map(Path, paths):
        done = p.parent.parent / "done" / p.name.replace("jobs_", "done_")
        out += attach_job_times(_read_jsonl(p), _read_jsonl(done))
    return latest_valid_jobs(out)


KEYS = ["config", "protocol", "task", "variant", "test_condition"]


def main_view(df: pd.DataFrame) -> pd.DataFrame:
    """Per tabelle e JSON compatto: delle varianti BN-α tiene solo n0 scelto in validation (n0sel) e
    la ristima pura (n00); le permutazioni restano solo nei file completi."""
    if df.empty:
        return df
    v = df.variant.astype(str)
    bn = v.str.contains("bnalpha_")
    keep = ~bn | v.str.endswith("_n0sel") | (v.str.endswith("_n00") & ~v.str.contains("_first"))
    return df[keep & ~v.str.contains(r"(?:^|_)perm\d+", regex=True)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per gruppo: media su soggetti della BA mediata su seed (e fold); sd tra soggetti; sd tra seed."""
    if df.empty:
        return df
    rows = []
    for key, g in df.groupby(KEYS):
        per_subj = g.groupby("subject")[["ba_win", "ba_trial", "kappa", "f1w"]].mean()
        per_seed = g.groupby("seed")["ba_win"].mean()
        rows.append({**dict(zip(KEYS, key)), "n_subj": len(per_subj), "n_seeds": g.seed.nunique(),
                     "n_folds": g.fold.nunique(),
                     "ba_win": per_subj.ba_win.mean(), "ba_win_sd_subj": per_subj.ba_win.std(),
                     "ba_win_sd_seed": per_seed.std() if len(per_seed) > 1 else np.nan,
                     "ba_trial": per_subj.ba_trial.mean(), "kappa": per_subj.kappa.mean(),
                     "f1w": per_subj.f1w.mean()})
    return pd.DataFrame(rows)


def markdown_table(s: pd.DataFrame, max_rows: int = 200) -> str:
    if s.empty:
        return "(nessun risultato)"
    # le condizioni tk{k} dei protocolli con clip condivise servono solo alla scomposizione
    s = s[~(s.test_condition.astype(str).str.startswith("tk") & s.protocol.isin(["split_fm", "lnso10"]))]
    s = s.sort_values(KEYS)
    lines = ["| config | protocollo | task | variante | condizione | sogg. | seed | BA finestra | BA trial | κ | F1w |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.head(max_rows).iterrows():
        sd_seed = "" if pd.isna(r.ba_win_sd_seed) else f" (seed sd {r.ba_win_sd_seed:.3f})"
        lines.append(f"| {r.config} | {r.protocol} | {r.task} | {r.variant} | {r.test_condition} | {r.n_subj} | "
                     f"{r.n_seeds} | {r.ba_win:.4f} ± {r.ba_win_sd_subj:.3f}{sd_seed} | {r.ba_trial:.4f} | "
                     f"{r.kappa:.4f} | {r.f1w:.4f} |")
    return "\n".join(lines)


def gate1(jobs: list[dict]) -> dict:
    """Gate 1: cbramod_ft_official su split_fm, BA a livello di finestra *pooled* (come il paper)."""
    vals = [j["pooled_all"]["ba_win"] for j in jobs
            if j.get("config") == "cbramod_ft_official" and "__split_fm__" in j["job_id"]
            and j["job_id"].endswith("__emo9__base") and "pooled_all" in j]
    if not vals:
        return {"status": "non ancora eseguito"}
    m = float(np.mean(vals))
    return {"n_seeds": len(vals), "ba_pooled_mean": round(m, 4), "per_seed": [round(v, 4) for v in vals],
            "target": GATE1_TARGET, "status": "PASSATO" if abs(m - GATE1_TARGET) <= GATE1_TOL else
            ("NON PASSATO" if len(vals) >= 3 else "parziale")}


GATE_R_TARGET, GATE_R_RANGE, GATE_R1 = 0.5646, (0.548, 0.581), (0.422, 0.028)


def gate_r(jobs: list[dict]) -> dict:
    """Gate R (Sessione 02): REVE-Base su split_fm, BA pooled per segmento all'epoca con la migliore BA di
    validation della fase FT (R2/R3); R1 = solo linear probing, da confrontare con 0.422 +- 0.028."""
    def vals(cfg):
        return {j["job_id"].split("__")[3][1:]: j["pooled_all"]["ba_win"] for j in jobs   # "...__f0__s42__..."
                if j.get("config") == cfg and "__split_fm__" in j["job_id"] and j["job_id"].endswith("__emo9__base")
                and "pooled_all" in j}
    r1, r2 = vals("reve_base_lp"), vals("reve_base")
    out = {"R1": {"per_seed": {k: round(v, 4) for k, v in r1.items()}, "target": GATE_R1}}
    if r1:
        m1 = float(np.mean(list(r1.values())))
        out["R1"]["mean"] = round(m1, 4)
        out["R1"]["diagnosi"] = ("compatibile" if abs(m1 - GATE_R1[0]) <= 2 * GATE_R1[1] else
                                 "vicino a 0.24: c'è un pooling" if abs(m1 - 0.24) < 0.05 else
                                 "vicino al caso: controlla pesi, scala, ordine dei canali, etichette" if m1 < 0.15 else
                                 "fuori intervallo")
    if not r2:
        return {**out, "status": "non ancora eseguito"}
    g = {s: v for s, v in r2.items() if int(s) in range(42, 47)}
    g10 = {s: v for s, v in r2.items() if int(s) in range(42, 52)}
    m = float(np.mean(list(g.values()))) if g else float("nan")
    out["R2"] = {"n_seeds": len(g), "mean": round(m, 4), "sd": round(float(np.std(list(g.values()), ddof=1)), 4)
                 if len(g) > 1 else None, "per_seed": {k: round(v, 4) for k, v in g.items()},
                 "range": GATE_R_RANGE, "target": GATE_R_TARGET}
    if len(g10) > len(g):
        out["R3"] = {"n_seeds": len(g10), "mean": round(float(np.mean(list(g10.values()))), 4)}
    out["status"] = ("PASSATO" if GATE_R_RANGE[0] <= m <= GATE_R_RANGE[1] else "NON PASSATO") if len(g) >= 5 else "parziale"
    return out


def timing(jobs: list[dict]) -> dict:
    """Minuti per job e per fase, epoche effettive ed epoca migliore (Sessione 02)."""
    out = {}
    for j in jobs:
        if not j.get("phase_minutes"):
            continue
        d = out.setdefault(j["config"], {"n": 0, "minutes": [], "phase_minutes": {}, "phase_epochs": {}, "best_epoch": []})
        d["n"] += 1
        d["minutes"].append(j.get("minutes", np.nan))
        d["best_epoch"].append(j.get("best_epoch"))
        for k, v in j["phase_minutes"].items():
            d["phase_minutes"].setdefault(k, []).append(v)
        for k, v in (j.get("phase_epochs") or {}).items():
            d["phase_epochs"].setdefault(k, []).append(v)
    return {c: {"n": d["n"], "min_job": round(float(np.nanmean(d["minutes"])), 1),
                "min_fase": {k: round(float(np.mean(v)), 1) for k, v in d["phase_minutes"].items()},
                "epoche": {k: round(float(np.mean(v)), 1) for k, v in d["phase_epochs"].items()},
                "epoca_migliore": round(float(np.mean([b for b in d["best_epoch"] if b is not None] or [np.nan])), 1)}
            for c, d in out.items()}


def _r3(v) -> list:
    """Arrotonda a 3 decimali; NaN -> null (JSON valido)."""
    return [None if not np.isfinite(x) else round(float(x), 3) for x in np.asarray(v, float)]


def compact_json(df: pd.DataFrame, jobs: list[dict], account: str) -> dict:
    """Medie per gruppo + punteggi per soggetto (mediati su seed e fold), arrotondati a 3 decimali."""
    out = {"v": 1, "account": account, "version": kaggle_env.code_version(), "groups": [], "gate1": gate1(jobs)}
    if not df.empty:
        for key, g in df.groupby(KEYS):
            ps = g.groupby("subject")[["ba_win", "ba_trial"]].mean()
            out["groups"].append({"k": list(key), "n_seeds": int(g.seed.nunique()), "subj": ps.index.tolist(),
                                  "bw": _r3(ps.ba_win.values), "bt": _r3(ps.ba_trial.values)})
    extra = {}
    for j in jobs:
        for k in ("clip_probe", "clip_probe_token_mean", "bnalpha_n0_selected", "perm_all", "perm_unseen",
                  "testperm_all", "testperm_unseen"):
            if k in j:
                extra.setdefault(j["job_id"], {})[k] = j[k]
    for j in jobs:          # REVE: revisione dei pesi e varianti di §R8 scelte (una volta per job)
        if str(j.get("config", "")).startswith("reve") and (j.get("pretrained") or j.get("variants")):
            extra.setdefault(j["job_id"], {}).update(
                {"pesi": {k: (j.get("pretrained") or {}).get(k) for k in ("revision", "sha256", "strict")},
                 "varianti": j.get("variants"), "best_epoch": j.get("best_epoch"),
                 "phase_best": j.get("phase_best")})
    out["job_extra"] = extra
    out["gate_r"] = gate_r(jobs)
    out["timing"] = timing(jobs)
    hpo = {}
    for p in (kaggle_env.state_dir() / "hpo_best").glob("*.json"):
        hpo[p.stem] = json.loads(p.read_text())
    out["hpo_best"] = hpo
    return out


def shortcut_summary(df: pd.DataFrame) -> dict:
    from .eval.shortcuts import delta_emg, delta_stim, stim_decomposition
    out = {"stim": {}, "emg": {}, "decomposition": {}}
    if df.empty:
        return out
    tasks = sorted(df.task.unique()) if "task" in df.columns else ["emo9"]
    for c in sorted(df.config.unique()):
        for t in tasks:
            sfx = "" if t == "emo9" else f"|{t}"
            for p in ("split_fm_lco", "lnso10_lco"):
                d = delta_stim(df, c, p, task=t)
                if d:
                    out["stim"][f"{c}|{p}{sfx}"] = {k: (np.round(v, 3).tolist() if isinstance(v, list) and k != "subjects"
                                                        else v) for k, v in d.items()}
            for bp in ("split_fm", "lnso10"):
                dec = stim_decomposition(df, c, bp, task=t)
                if dec:
                    out["decomposition"][f"{c}|{bp}{sfx}"] = dec
        for ep in ("split_fm", "lnso10"):
            e = delta_emg(df, c, protocol=ep)
            if e and len(e) > 2:
                out["emg"][c if ep == "split_fm" else f"{c}|{ep}"] = {k: (np.round(v, 3).tolist() if isinstance(v, list) and k != "subjects" else v)
                                 for k, v in e.items()}
    return out


def shortcut_lines(sc: dict) -> list[str]:
    lines = []
    for k, d in sc.get("stim", {}).items():
        line = (f"- Δstim {k}: viste {d['seen_mean']:.3f}, non viste {d['unseen_mean']:.3f}, grezzo {d['raw_mean']:.4f}")
        if "corr_mean" in d:
            line += (f"; permutate viste {d['seen_perm_mean']:.3f} / non viste {d['unseen_perm_mean']:.3f}; "
                     f"memorizzazione {d['memorization_mean']:.4f}; corretto (vs caso) {d['corr_chance_mean']:.4f}; "
                     f"corretto (prompt, distorto) {d['corr_mean']:.4f}")
        lines.append(line + f" ({len(d['subjects'])} soggetti, {d['n_perm_runs']} run permutati)")
    from .eval.shortcuts import decomposition_lines
    for k, dec in sc.get("decomposition", {}).items():
        lines += decomposition_lines(dec, k)
    for c, e in sc.get("emg", {}).items():
        ks = [k for k in e if k.startswith("delta_")]
        lines.append(f"- ΔEMG {c} (BA piena - BA variante): " + ", ".join(f"{k[6:]} {e[k]:.4f}" for k in ks))
    return lines


def print_summary(account: str = "account", max_rows: int = 200) -> dict:
    df = load_rows()
    jobs = load_jobs()
    s = summarize(main_view(df))
    print("\n## Riepilogo risultati\n")
    print(markdown_table(s, max_rows))
    g1 = gate1(jobs)
    print(f"\n**Gate 1 (riproduzione CBraMod, BA pooled finestra, target {GATE1_TARGET} ± {GATE1_TOL})**: {g1}")
    for line in shortcut_lines(shortcut_summary(df)):
        print(line)
    cj = compact_json(main_view(df), jobs, account)
    cj["shortcuts"] = shortcut_summary(df)
    txt = json.dumps(cj, separators=(",", ":"))
    print("\n## JSON compatto (copialo tutto in chat)\n")
    print("PIANOB_JSON_BEGIN")
    print(txt)
    print("PIANOB_JSON_END")
    kaggle_env.write_json(kaggle_env.working_dir() / f"summary_{account}.json", cj)
    return cj


def make_results_zip(account: str) -> Path:
    st = kaggle_env.state_dir()
    out = kaggle_env.working_dir() / f"results_{account}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("results", "done", "hpo_best", "bench", "logs", "optuna", "selfreport"):
            for f in (st / sub).rglob("*"):
                if f.is_file():
                    z.write(f, f"{kaggle_env.STATE_DIRNAME}/{f.relative_to(st)}")
        for f in [*(st / "artifacts" / "preds").glob("*.npz"), *(st / "artifacts" / "srpreds").glob("*.npz")]:
            z.write(f, f"{kaggle_env.STATE_DIRNAME}/{f.relative_to(st)}")
        s = kaggle_env.working_dir() / f"summary_{account}.json"
        if s.exists():
            z.write(s, s.name)
    print(f"[zip] {out} ({out.stat().st_size / 2**20:.1f} MB)")
    return out
