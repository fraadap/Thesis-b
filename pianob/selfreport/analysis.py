"""Analisi della Sessione 03: unione dei fold, tabelle 2 x 2, rianalisi delle predizioni salvate, terza famiglia.

Sorgenti: cartelle di stato (pianob_state) o zip di risultati, con
- artifacts/srpreds/<job>.npz : predizioni per trial dei task da self-report (H3, H4, notebook I);
- artifacts/preds/<job>.npz   : probabilità per finestra a 9 classi dei job emo9 (rianalisi H2, zero training);
- selfreport/selfreport.parquet: tabella dei voti (passo 0).

Unione dei fold (§3.1), per seed: in leave-clip-out ogni coppia (soggetto, clip) è non vista in esattamente un fold
(asserzione); nelle clip viste è la media dei fold in cui è vista (e, per robustezza, un solo fold scelto con
SEED_SINGLE). Le metriche si calcolano sulla matrice soggetti x clip della condizione, per seed; si riporta la
media sui seed, mentre IC e p si calcolano sulla predizione mediata sui seed (dichiarato).
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import faced_meta as FM
from ..data.splits import heldout_sets, make_split
from ..eval.stats import holm, paired_test
from . import metrics as M
from .targets import SR_TASKS

SEED_SINGLE = 4242
ITEM_OF = {"valence": "valence", "arousal": "arousal"}
JOB_RE = re.compile(r"^(?P<config>.+?)__(?P<protocol>[a-z0-9_]+?)__f(?P<fold>\d+)__s(?P<seed>\d+)__(?P<task>[a-z0-9_]+?)__(?P<variant>.+)$")


# ------------------------------------------------------------------ lettura
def iter_npz(sources, sub: str):
    """(job_id, dict di array) da cartelle di stato o zip; `sub` = 'srpreds' oppure 'preds'."""
    seen = set()
    for src in map(Path, sources):
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                for n in z.namelist():
                    if f"/artifacts/{sub}/" in n and n.endswith(".npz"):
                        jid = Path(n).stem
                        if jid not in seen:
                            seen.add(jid)
                            with np.load(io.BytesIO(z.read(n)), allow_pickle=False) as d:
                                yield jid, {k: d[k] for k in d.files}
        else:
            for f in sorted(src.rglob(f"artifacts/{sub}/*.npz")):
                if f.stem not in seen:
                    seen.add(f.stem)
                    with np.load(f, allow_pickle=False) as d:
                        yield f.stem, {k: d[k] for k in d.files}


def find_votes(sources) -> pd.DataFrame | None:
    for src in map(Path, sources):
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                for n in z.namelist():
                    if n.endswith("selfreport/selfreport.parquet"):
                        return pd.read_parquet(io.BytesIO(z.read(n)))
        else:
            for f in src.rglob("selfreport/selfreport.parquet"):
                return pd.read_parquet(f)
    return None


def parse_job(jid: str) -> dict | None:
    m = JOB_RE.match(jid)
    return {**m.groupdict(), "fold": int(m["fold"]), "seed": int(m["seed"])} if m else None


# ------------------------------------------------------------------ unione dei fold
def merge_folds(recs: list[dict], key: str = "p") -> dict:
    """recs: [{fold, protocol, subject, clip, cond, <key>}] dello stesso (config, protocollo, task, variante, seed).
    Ritorna cond -> DataFrame(subject, clip, value). Condizioni: all, seen, seen_single, unseen, tk0..2."""
    by = defaultdict(list)
    for r in recs:
        for cond in np.unique(r["cond"]):
            m = r["cond"] == cond
            by[str(cond)].append(pd.DataFrame({"subject": r["subject"][m], "clip": r["clip"][m],
                                               "v": np.asarray(r[key])[m], "fold": r["fold"]}))
    out = {}
    for cond, dfs in by.items():
        df = pd.concat(dfs, ignore_index=True)
        if cond == "unseen":
            dup = df.duplicated(["subject", "clip"]).sum()
            assert dup == 0, f"unseen: {dup} coppie (soggetto, clip) non viste in più di un fold"
            out[cond] = df[["subject", "clip", "v"]]
        else:
            out[cond] = df.groupby(["subject", "clip"], as_index=False)["v"].mean()
            if cond == "seen":
                rng = np.random.default_rng(SEED_SINGLE)
                df = df.assign(r=rng.random(len(df))).sort_values("r")
                out["seen_single"] = df.drop_duplicates(["subject", "clip"])[["subject", "clip", "v"]]
    if "all" in out:      # T_k dentro le clip condivise (stessa definizione della scomposizione)
        for k, clips in heldout_sets().items():
            out[f"tk{k}"] = out["all"][out["all"].clip.isin(clips)]
    return out


def to_matrices(dfv: pd.DataFrame, Zfull: np.ndarray):
    """Matrice P e Z (soggetti completi della condizione, con voto definito)."""
    P = dfv.pivot(index="subject", columns="clip", values="v")
    P = P.dropna(axis=0, how="any")
    Z = Zfull[P.index.to_numpy()][:, P.columns.to_numpy()]
    ok = ~np.isnan(Z).any(1)
    return P.to_numpy()[ok], Z[ok], P.index.to_numpy()[ok], P.columns.to_numpy()


def zmatrix(votes: pd.DataFrame, item: str) -> np.ndarray:
    Z = np.full((int(votes.subject.max()) + 1, FM.N_CLIPS), np.nan)
    Z[votes["subject"].to_numpy(int), votes["clip"].to_numpy(int)] = votes[f"{item}_z"].to_numpy(float)
    return Z


# ------------------------------------------------------------------ metriche di un gruppo
def ceiling(Z: np.ndarray) -> float:
    """Quota della varianza del voto z nella componente residua (interazione + errore): tetto delle metriche
    entro clip con un solo voto per coppia."""
    return float(M.anova2(Z)["share_residual"])


def group_metrics(P_by_seed: list[np.ndarray], Z: np.ndarray, n_boot: int = 2000, n_perm: int = 5000,
                  seed: int = 0, labels: np.ndarray | None = None) -> dict:
    per_seed = [M.all_metrics(P, Z) for P in P_by_seed]
    Pm = np.mean(P_by_seed, 0)
    out = {k: float(np.mean([d[k] for d in per_seed])) for k in M.METRICS}
    out["n_subjects"], out["n_clips"], out["n_seeds"] = int(Z.shape[0]), int(Z.shape[1]), len(P_by_seed)
    if n_boot:
        out["ci_crossed"] = M.bootstrap_ci(Pm, Z, n=n_boot, seed=seed, crossed=True)
        out["ci_subjects"] = M.bootstrap_ci(Pm, Z, n=n_boot, seed=seed + 1, crossed=False)
    if n_perm:
        out["perm"] = M.perm_test(Pm, Z, n=n_perm, seed=seed)
        out["perm_recentred"] = M.perm_test(Pm, Z, n=n_perm, seed=seed, mode="recenter")
        rng = np.random.default_rng(seed + 2)               # nullo della correlazione tra clip: clip permutate
        nb = [float(M.between_clip(Pm[:, rng.permutation(Pm.shape[1])], Z)) for _ in range(min(n_perm, 2000))]
        out["p_between_clip"] = (1 + sum(v >= out["between_clip"] for v in nb)) / (1 + len(nb))
    out["ceiling_residual_share"] = ceiling(Z)
    out["per_clip_within"] = M.per_clip_within(Pm, Z).round(3).tolist()
    out["within_clip_pearson"] = float(M.within_clip(Pm, Z, method="pearson"))
    if labels is not None:            # BA binaria per soggetto: predizione = log-odds > 0
        out["ba_bin"] = float(np.nanmean([_ba(Pm[i] > 0, labels[i]) for i in range(len(Pm))]))
    return out


def _ba(pred, lab) -> float:
    lab = np.asarray(lab, float)
    ok = lab >= 0
    if ok.sum() == 0 or len(np.unique(lab[ok])) < 2:
        return np.nan
    return float(np.mean([np.mean(pred[ok][lab[ok] == k] == k) for k in (0, 1)]))


# ------------------------------------------------------------------ predizioni dei task da self-report
def load_sr_groups(sources) -> dict:
    """(config, protocollo, task, variante) -> seed -> lista di record per fold."""
    groups = defaultdict(lambda: defaultdict(list))
    for jid, d in iter_npz(sources, "srpreds"):
        meta = parse_job(jid)
        if meta is None:
            continue
        rec = {**meta, **d}
        groups[(meta["config"], meta["protocol"], meta["task"], meta["variant"])][meta["seed"]].append(rec)
    return groups


def bin_label_matrix(votes, task, n_slots) -> np.ndarray:
    from .targets import label_matrix
    spec = SR_TASKS[task]
    if spec["kind"] not in ("bin", "bin_strict"):
        return None
    return label_matrix(task, votes, n_slots)


def analyze_sr(sources, votes, n_boot=2000, n_perm=5000, conds=("all", "seen", "unseen", "seen_single",
                                                                  "tk0", "tk1", "tk2"), log=print) -> list[dict]:
    out = []
    n_slots = int(votes.subject.max()) + 1
    for (config, protocol, task, variant), by_seed in sorted(load_sr_groups(sources).items()):
        spec = SR_TASKS.get(task)
        if spec is None or spec["kind"] == "argmax8":
            continue
        Zfull = zmatrix(votes, spec["item"])
        L = bin_label_matrix(votes, task, n_slots)
        merged = {s: merge_folds(recs) for s, recs in by_seed.items()}
        perm_merged = {}
        for s, recs in by_seed.items():          # modelli riaddestrati con i residui permutati (feature)
            if all("perm_p" in r for r in recs):
                K = recs[0]["perm_p"].shape[1]
                perm_merged[s] = [merge_folds([{**r, "pk": r["perm_p"][:, k]} for r in recs], key="pk")
                                  for k in range(K)]
        for cond in conds:
            mats = [to_matrices(m[cond], Zfull) for m in merged.values() if cond in m]
            if not mats:
                continue
            subj = sorted(set.intersection(*[set(x[2]) for x in mats]))
            clips = mats[0][3]
            Ps = [x[0][np.isin(x[2], subj)] for x in mats]
            Z = mats[0][1][np.isin(mats[0][2], subj)]
            lab = L[np.asarray(subj)][:, clips] if L is not None else None
            g = group_metrics(Ps, Z, n_boot, n_perm, labels=lab)
            if perm_merged:
                pv = {"resid_corr": [], "within_clip": []}
                for pm_list in perm_merged.values():
                    for pm in pm_list:
                        if cond in pm:
                            Pk, Zk, sk, _ = to_matrices(pm[cond], Zfull)
                            for mname in pv:
                                pv[mname].append(float(M.METRICS[mname](Pk, Zk)))
                g["train_perm"] = {m: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                                       "p": (1 + sum(x >= g[m] for x in v)) / (1 + len(v)), "n": len(v)}
                                   for m, v in pv.items() if v}
            out.append({"config": config, "protocol": protocol, "task": task, "variant": variant, "cond": cond, **g})
            log(f"[S03] {config} {protocol} {task} {variant} {cond}: residui {g['resid_corr']:+.3f}, "
                f"entro clip {g['within_clip']:+.3f}, tra clip {g['between_clip']:+.3f}")
    return out


# ------------------------------------------------------------------ rianalisi delle predizioni a 9 classi (H2)
def _abar(Z: np.ndarray, split) -> np.ndarray:
    """Media di z per categoria sui soli soggetti di training e sulle loro clip di training."""
    tr = np.asarray(split.train_subjects)
    clips = np.asarray(split.train_clips)
    cat = np.asarray(FM.CLIP_LABELS)
    out = np.zeros(len(FM.CATEGORIES))
    for c in range(len(FM.CATEGORIES)):
        cc = clips[cat[clips] == c]
        out[c] = np.nanmean(Z[np.ix_(tr, cc)]) if len(cc) else np.nan
    return out


def reanalyze_emo9(sources, votes, items=("valence", "arousal"), n_boot=2000, n_perm=5000,
                   conds=("all", "seen", "unseen", "seen_single", "tk0", "tk1", "tk2"), log=print) -> dict:
    """p(s,i) = sum_c P(c | finestre) * abar_c (abar dai soggetti di training del job); memorizzatore puro di
    riferimento p = abar_{categoria(i)}; trial concordanti contro discordanti (accuratezza a 9 classi)."""
    conc = np.full((int(votes.subject.max()) + 1, FM.N_CLIPS), np.nan)
    conc[votes["subject"].to_numpy(int), votes["clip"].to_numpy(int)] = votes["concordant"].to_numpy(float)
    groups = defaultdict(lambda: defaultdict(list))
    for jid, d in iter_npz(sources, "preds"):
        meta = parse_job(jid)
        if meta is None or meta["task"] != "emo9" or d["probs"].shape[1] != 9:
            continue
        groups[(meta["config"], meta["protocol"], meta["variant"])][meta["seed"]].append({**meta, **d})
    res = {"metrics": [], "concordance": [], "memorizer": []}
    for item in items:
        Zfull = zmatrix(votes, item)
        for (config, protocol, variant), by_seed in sorted(groups.items()):
            merged, accm, memm = {}, {}, {}
            for s, recs in by_seed.items():
                tr_recs, acc_recs, mem_recs = [], [], []
                for r in recs:
                    sp = make_split(protocol, r["fold"], list(range(FM.N_CLIPS)), None, clip_labels=FM.CLIP_LABELS,
                                    draw=0)
                    ab = _abar(Zfull, sp)
                    key = r["subject"].astype(np.int64) * 1000 + r["clip"]
                    uk, inv = np.unique(key, return_inverse=True)
                    Pbar = np.zeros((len(uk), 9))
                    np.add.at(Pbar, inv, r["probs"].astype(np.float64))
                    Pbar /= np.bincount(inv)[:, None]
                    ss, cc = uk // 1000, uk % 1000
                    cond = np.where(np.isin(cc, sp.unseen_clips), "unseen", "seen") if sp.is_lco \
                        else np.full(len(uk), "all")
                    base = {"subject": ss, "clip": cc, "cond": cond, "fold": r["fold"]}
                    tr_recs.append({**base, "p": Pbar @ ab})
                    mem_recs.append({**base, "p": ab[np.asarray(FM.CLIP_LABELS)[cc]]})
                    acc_recs.append({**base, "p": (Pbar.argmax(1) == np.asarray(FM.CLIP_LABELS)[cc]).astype(float)})
                merged[s], memm[s], accm[s] = merge_folds(tr_recs), merge_folds(mem_recs), merge_folds(acc_recs)
            for cond in conds:
                mats = [to_matrices(m[cond], Zfull) for m in merged.values() if cond in m]
                if not mats:
                    continue
                subj = sorted(set.intersection(*[set(x[2]) for x in mats]))
                Ps = [x[0][np.isin(x[2], subj)] for x in mats]
                Z = mats[0][1][np.isin(mats[0][2], subj)]
                g = group_metrics(Ps, Z, n_boot if variant == "base" else 0, n_perm if variant == "base" else 0)
                res["metrics"].append({"config": config, "protocol": protocol, "variant": variant, "item": item,
                                       "cond": cond, **g})
                mm = to_matrices(next(iter(memm.values()))[cond], Zfull)
                res["memorizer"].append({"config": config, "protocol": protocol, "variant": variant, "item": item,
                                         "cond": cond, **M.all_metrics(mm[0], mm[1])})
                if item == items[0]:     # concordanti contro discordanti, accuratezza per soggetto (primo seed)
                    A, _, sa, ca = to_matrices(next(iter(accm.values()))[cond], np.zeros_like(Zfull))
                    C = conc[sa][:, ca]
                    acc_c = [np.nanmean(np.where(C[i] == 1, A[i], np.nan)) for i in range(len(A))]
                    acc_d = [np.nanmean(np.where(C[i] == 0, A[i], np.nan)) for i in range(len(A))]
                    res["concordance"].append({"config": config, "protocol": protocol, "variant": variant,
                                               "cond": cond, "acc_concordant": float(np.nanmean(acc_c)),
                                               "acc_discordant": float(np.nanmean(acc_d)),
                                               "test": paired_test(acc_c, acc_d)})
                log(f"[H2] {config} {protocol} {variant} {item} {cond}: residui {g['resid_corr']:+.3f}, "
                    f"entro clip {g['within_clip']:+.3f}, tra clip {g['between_clip']:+.3f}")
    return res


# ------------------------------------------------------------------ terza famiglia
def third_family(sr_results: list[dict], rows: pd.DataFrame | None = None, config: str = "de_asym_lr_zsub",
                 protocol: str = "lnso10_lco") -> dict:
    """Holm sulle 4 ipotesi di §6 (unilaterali), modello fissato a priori: DE + z-score su lnso10_lco."""
    def get(task):
        return next((r for r in sr_results if r["config"] == config and r["protocol"] == protocol
                     and r["task"] == task and r["variant"] == "base" and r["cond"] == "unseen"), None)
    out = {}
    v, a = get("sr_val_bin"), get("sr_aro_bin")
    if v and "perm" in v:
        out["T1_valenza_entro_clip_nonviste"] = {"stat": v["resid_corr"], "p": v["perm"]["resid_corr"]["p"],
                                                 "ci_crossed": v.get("ci_crossed", {}).get("resid_corr")}
        out["T3_valenza_tra_clip_nonviste"] = {"stat": v["between_clip"], "p": v["p_between_clip"],
                                               "ci_crossed": v.get("ci_crossed", {}).get("between_clip")}
    if a and "perm" in a:
        out["T2_arousal_entro_clip_nonviste"] = {"stat": a["resid_corr"], "p": a["perm"]["resid_corr"]["p"],
                                                 "ci_crossed": a.get("ci_crossed", {}).get("resid_corr")}
    if rows is not None and len(rows):
        sel = (rows.config == config) & (rows.protocol == protocol) & (rows.variant == "base") & \
              (rows.test_condition == "unseen")
        sr = rows[sel & (rows.task == "sr_val_bin")].groupby("subject")["ba_win"].mean()
        cat = rows[sel & (rows.task == "val2")].groupby("subject")["ba_win"].mean()
        idx = sr.index.intersection(cat.index)
        if len(idx):
            t = paired_test(sr[idx].values, cat[idx].values, alternative="greater")
            out["T4_label_provata_vs_categoria_nonviste"] = {"stat": t["mean_diff"], "p": t["p"], "n": t["n"],
                                                             "ci95": t.get("ci95"),
                                                             "nota": "BA di ciascun target contro la propria label"}
    adj = holm({k: v["p"] for k, v in out.items()})
    for k in out:
        out[k]["p_holm"] = adj[k]
    return out


# ------------------------------------------------------------------ descrittive dei voti (H1)
def descriptives(votes: pd.DataFrame, meta: dict | None = None, n_boot: int = 1000) -> dict:
    out = {"items": {}}
    for item in ["valence", "arousal", "liking", "familiarity"]:
        raw = votes.pivot(index="subject", columns="clip", values=item).dropna().to_numpy()
        Z = votes.pivot(index="subject", columns="clip", values=f"{item}_z").dropna().to_numpy()
        a = M.anova2(raw)
        out["items"][item] = {"raw": {k: float(v) for k, v in a.items()}, "raw_ci": M.anova2_ci(raw, n_boot),
                              "z_eta2_clip": float(M.anova2(Z)["eta2_clip"]),
                              "z_residual_share": float(M.anova2(Z)["share_residual"]),
                              "split_half": M.split_half(raw, 1000), "n_subjects": int(len(raw))}
    E = {it: M.resid2(votes.pivot(index="subject", columns="clip", values=f"{it}_z").dropna().to_numpy())
         for it in ["valence", "arousal", "liking", "familiarity"]}
    n = min(len(v) for v in E.values())
    out["within_clip_item_corr"] = {f"{a}~{b}": float(np.corrcoef(E[a][:n].ravel(), E[b][:n].ravel())[0, 1])
                                    for a in E for b in E if a < b}
    out["mde"] = {"lnso10 / lnso10_lco (123 soggetti x 28 clip)": {"residui": M.mde_resid_corr(123, 28),
                                                                    "entro_clip": M.mde_within_clip(123, 28)},
                  "split_fm / split_fm_lco (23 soggetti x 28 clip)": {"residui": M.mde_resid_corr(23, 28),
                                                                       "entro_clip": M.mde_within_clip(23, 28)},
                  "tk (23 soggetti x 9-10 clip)": {"residui": M.mde_resid_corr(23, 9),
                                                   "entro_clip": M.mde_within_clip(23, 9)}}
    if meta:
        out["meta"] = {k: meta.get(k) for k in ("alignment", "columns", "qc", "concordance", "stimuli_info",
                                                "valence_pos_gt_neg_fraction", "files", "structure_first3")}
        out["meta"]["constant_items"] = meta.get("derived", {}).get("constant")
        ties = meta.get("derived", {}).get("ties_at_median", {})
        out["meta"]["ties_at_median_mean"] = {k: float(np.mean(list(v.values()))) for k, v in ties.items() if v}
    return out


# ------------------------------------------------------------------ rapporti markdown
def _f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}" if isinstance(x, float) else str(x)


def stage1_markdown(desc: dict) -> str:
    md = ["# Sessione 03 — Rapporto della fase 1 (voti da self-report)\n"]
    m = desc.get("meta") or {}
    if m:
        al = m.get("alignment") or {}
        md += ["## Struttura dei file e allineamento\n",
               f"- File: {m.get('files')}",
               f"- Primi soggetti: `{json.dumps(m.get('structure_first3'), default=str)[:1500]}`",
               f"- Allineamento scelto: **{al.get('alignment')}** (concordanza con vid {al.get('concordance_vid', float('nan')):.3f}, "
               f"canonica {al.get('concordance_canonical', float('nan')):.3f}; vid identità ovunque: {al.get('vid_identity_everywhere')})",
               f"- Stimuli_info: {m.get('stimuli_info')}",
               f"- Controlli: {m.get('qc')}; item costanti per soggetto: {m.get('constant_items')}; "
               f"pareggi medi alla mediana: {m.get('ties_at_median_mean')}",
               f"- Valenza più alta sulle clip positive per il {100 * (m.get('valence_pos_gt_neg_fraction') or 0):.0f}% dei soggetti\n",
               "## Ordine delle colonne (media item x categoria)\n"]
        cols = m.get("columns") or {}
        tab = pd.DataFrame(cols.get("table", {}))
        if len(tab):
            try:
                md.append(tab.to_markdown())
            except ImportError:            # tabulate assente
                md.append("```\n" + tab.to_string() + "\n```")
        md.append(f"\nItem con il massimo sulla propria categoria: {cols.get('n_matches')}/8 "
                  f"({cols.get('item_max_on_own_category')})\n")
        c = m.get("concordance") or {}
        md.append(f"## Concordanza argmax8 / categoria\n\nmedia {c.get('mean', float('nan')):.3f}, mediana "
                  f"{c.get('median', float('nan')):.3f} (caso 0.125); soggetti con concordanza <= 1/8: {c.get('flagged_le_1_8')}\n")
    md.append("## Scomposizione della varianza e affidabilità\n")
    md.append("| item | soggetto | clip | residuo | eta2 clip (z) | quota residua (z) = tetto | ICC(2,1) | ICC(2,k) | split-half |\n|---|---|---|---|---|---|---|---|---|")
    for it, d in desc["items"].items():
        r = d["raw"]
        md.append(f"| {it} | {r['share_subject']:.2f} | {r['share_clip']:.2f} | {r['share_residual']:.2f} | "
                  f"{d['z_eta2_clip']:.2f} | {d['z_residual_share']:.2f} | {r['icc2_1']:.2f} | {r['icc2_k']:.2f} | "
                  f"{d['split_half']:.2f} |")
    md.append(f"\nCorrelazioni tra item sui residui entro clip: {json.dumps({k: round(v, 3) for k, v in desc['within_clip_item_corr'].items()})}\n")
    md.append("## Effetto minimo rilevabile (unilaterale, alfa 0.05, potenza 0.8)\n")
    for k, v in desc["mde"].items():
        md.append(f"- {k}: residui a due vie r = {v['residui']:.3f}, entro clip r = {v['entro_clip']:.3f}")
    return "\n".join(md) + "\n"


def table_2x2(sr_results: list[dict]) -> list[str]:
    md = ["| modello | protocollo | target | variante | tra clip viste | tra clip non viste | entro clip viste (residui / entro clip) | entro clip non viste (residui / entro clip) | p non viste | tetto | permutato al training (entro clip, media) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    key = lambda r: (r["config"], r["protocol"], r["task"], r["variant"])
    groups = defaultdict(dict)
    for r in sr_results:
        groups[key(r)][r["cond"]] = r
    for k, d in sorted(groups.items()):
        seen = d.get("seen") or d.get("all")
        un = d.get("unseen")
        def cell(r):
            return "—" if r is None else f"{r['resid_corr']:+.3f} / {r['within_clip']:+.3f}"
        tp = (un or seen or {}).get("train_perm", {}).get("within_clip", {}).get("mean")
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {_f(seen and seen['between_clip'])} | "
                  f"{_f(un and un['between_clip'])} | {cell(seen)} | {cell(un)} | "
                  f"{(un or {}).get('perm', {}).get('resid_corr', {}).get('p', '—') if un else '—'} | "
                  f"{(un or seen)['ceiling_residual_share']:.2f} | {_f(tp)} |")
    return md


def session03_markdown(sr_results, h2=None, family=None, rows=None) -> list[str]:
    md = ["\n## Sessione 03 — label da self-report\n"]
    if family:
        md.append("### Terza famiglia (Holm su 4, aggiunta successiva; DE + z-score, lnso10_lco, unilaterali)\n")
        for k, v in family.items():
            md.append(f"- **{k}**: statistica {v['stat']:+.4f}, p {v['p']:.3g}, p Holm {v['p_holm']:.3g}"
                      + (f", IC incrociato {np.round(v['ci_crossed'], 3).tolist()}" if v.get("ci_crossed") else ""))
    if sr_results:
        md.append("\n### Tabelle 2 x 2 (componente tra / entro clip x clip viste / non viste)\n")
        md += table_2x2(sr_results)
    if rows is not None and len(rows):
        md.append("\n### BA binaria: label provata contro label di categoria (valenza)\n")
        md.append("| config | protocollo | condizione | BA sr_val_bin | BA val2 (categoria) |\n|---|---|---|---|---|")
        for (c, p, cond), g in rows[rows.task.isin(["sr_val_bin", "val2"]) & (rows.variant == "base")] \
                .groupby(["config", "protocol", "test_condition"]):
            a = g[g.task == "sr_val_bin"].groupby("subject").ba_win.mean().mean()
            b = g[g.task == "val2"].groupby("subject").ba_win.mean().mean()
            md.append(f"| {c} | {p} | {cond} | {a:.3f} | {b:.3f} |")
    if h2:
        md.append("\n### Rianalisi delle predizioni a 9 classi (H2), variante base\n")
        md.append("| config | protocollo | item | condizione | residui | entro clip | tra clip | p residui | memorizzatore entro clip |\n|---|---|---|---|---|---|---|---|---|")
        mem = {(r["config"], r["protocol"], r["variant"], r["item"], r["cond"]): r for r in h2.get("memorizer", [])}
        for r in h2["metrics"]:
            if r["variant"] != "base" or r["cond"] not in ("all", "seen", "unseen"):
                continue
            mm = mem.get((r["config"], r["protocol"], r["variant"], r["item"], r["cond"]), {})
            md.append(f"| {r['config']} | {r['protocol']} | {r['item']} | {r['cond']} | {r['resid_corr']:+.3f} | "
                      f"{r['within_clip']:+.3f} | {r['between_clip']:+.3f} | "
                      f"{r.get('perm', {}).get('resid_corr', {}).get('p', float('nan')):.3g} | {mm.get('within_clip', float('nan')):+.3f} |")
        md.append("\n**Trial concordanti contro discordanti (accuratezza a 9 classi per soggetto)**\n")
        for r in h2.get("concordance", []):
            if r["variant"] == "base" and r["cond"] in ("all", "seen", "unseen"):
                md.append(f"- {r['config']} {r['protocol']} {r['cond']}: concordanti {r['acc_concordant']:.3f}, "
                          f"discordanti {r['acc_discordant']:.3f}, p {r['test'].get('p', float('nan')):.3g}")
    return md


def run_all(sources, out_dir: Path, rows: pd.DataFrame | None = None, n_boot=2000, n_perm=5000, h2=True,
            log=print) -> dict:
    """Analisi completa su sorgenti (cartelle o zip); scrive results/selfreport/*.json e un markdown."""
    votes = find_votes(sources)
    if votes is None:
        log("[S03] tabella dei voti assente nelle sorgenti: analisi saltata")
        return {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = analyze_sr(sources, votes, n_boot, n_perm, log=log)
    h2r = reanalyze_emo9(sources, votes, n_boot=n_boot, n_perm=n_perm, log=log) if h2 else None
    fam = third_family(sr, rows)
    (out_dir / "metrics_sr.json").write_text(json.dumps(sr, indent=1, default=float))
    if h2r is not None:
        (out_dir / "reanalysis_h2.json").write_text(json.dumps(h2r, indent=1, default=float))
    (out_dir / "third_family.json").write_text(json.dumps(fam, indent=1, default=float))
    md = session03_markdown(sr, h2r, fam, rows)
    (out_dir / "report_selfreport.md").write_text("\n".join(md) + "\n")
    return {"sr": sr, "h2": h2r, "family": fam, "markdown": md}
