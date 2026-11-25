"""Analisi della Sessione 02 (ricetta a due fasi, pretraining su clip non viste, REVE contro CBraMod).

Seconda famiglia di analisi (dichiarata come aggiunta successiva; Holm solo su queste 4, le quattro ipotesi
primarie pre-registrate restano invariate):
  S1 pretraining sulle clip NON viste: cbramod_ft_official contro il da-zero, ricette scelte in validation
     (stessa regola di H1), bilaterale come H1;
  S2 CBraMod a due fasi sulle clip non viste contro il caso (Wilcoxon unilaterale, > 1/9);
  S3 REVE sulle clip non viste contro il caso (unilaterale, > 1/9);
  S4 dipendenza di REVE dal contenuto fuori 4-30 Hz: banda piena - 4-30 Hz > 0 (unilaterale), split_fm.
Tutto il resto è esplorativo.
"""
from __future__ import annotations

import numpy as np

from .shortcuts import delta_emg, delta_stim, stim_decomposition
from .stats import holm, paired_test

CHANCE9 = 1 / 9


def subj(df, config, protocol="split_fm", variant="base", cond="all", task="emo9", metric="ba_win"):
    d = df[(df.config == config) & (df.protocol == protocol) & (df.variant == variant) &
           (df.test_condition == cond) & (df.task == task)]
    return d.groupby("subject")[metric].mean() if not d.empty else None


def _pair(sa, sb, alternative="two-sided"):
    if sa is None or sb is None:
        return None
    idx = sa.index.intersection(sb.index)
    return paired_test(sa[idx].values, sb[idx].values, alternative=alternative) if len(idx) else None


def _vs_chance(s, chance=CHANCE9):
    return None if s is None else paired_test(s.values, np.full(len(s), chance), alternative="greater")


def _best_val(jobs, configs, protocol="split_fm"):
    sc = {}
    for j in jobs or []:
        if j.get("config") in configs and f"__{protocol}__" in j["job_id"] and j["job_id"].endswith("__emo9__base") \
                and "val_best" in j:
            sc.setdefault(j["config"], []).append(j["val_best"])
    return max(sc, key=lambda c: np.mean(sc[c])) if sc else None


def pretraining_pairs(jobs):
    pre = _best_val(jobs, ["cbramod_ft_official", "cbramod_ft_official_warmup"]) or "cbramod_ft_official"
    scr = _best_val(jobs, ["cbramod_scratch_ft", "cbramod_scratch_ft_warmup", "cbramod_scratch_ft_definit_warmup"]) \
        or "cbramod_scratch_ft_warmup"
    return {"principale (ricette scelte in validation)": (pre, scr),
            "ricetta identica (warm-up)": ("cbramod_ft_official_warmup", "cbramod_scratch_ft_warmup"),
            "ricetta a due fasi": ("cbramod_twophase_pre", "cbramod_twophase_scratch")}


def pretraining_on_unseen(df, jobs) -> dict:
    """Parte B: confronti appaiati per soggetto, separatamente su viste e non viste, + interazione."""
    out = {}
    for name, (a, b) in pretraining_pairs(jobs).items():
        r = {"coppia": f"{a} vs {b}"}
        for cond in ("seen", "unseen"):
            t = _pair(subj(df, a, "split_fm_lco", cond=cond), subj(df, b, "split_fm_lco", cond=cond))
            if t:
                r[cond] = t
        sa_s, sb_s = subj(df, a, "split_fm_lco", cond="seen"), subj(df, b, "split_fm_lco", cond="seen")
        sa_u, sb_u = subj(df, a, "split_fm_lco", cond="unseen"), subj(df, b, "split_fm_lco", cond="unseen")
        if all(x is not None for x in (sa_s, sb_s, sa_u, sb_u)):
            idx = sa_s.index.intersection(sb_s.index).intersection(sa_u.index).intersection(sb_u.index)
            inter = (sa_s[idx] - sb_s[idx]) - (sa_u[idx] - sb_u[idx])
            r["interazione (pre-scr)_viste - (pre-scr)_nonviste"] = paired_test(inter.values, np.zeros(len(idx)))
        if len(r) > 1:
            out[name] = r
    return out


def second_family(df, jobs) -> dict:
    res = {}
    pre, scr = pretraining_pairs(jobs)["principale (ricette scelte in validation)"]
    t = _pair(subj(df, pre, "split_fm_lco", cond="unseen"), subj(df, scr, "split_fm_lco", cond="unseen"))
    if t:
        res["S1_pretraining_nonviste"] = {**t, "coppia": f"{pre} vs {scr}"}
    t = _vs_chance(subj(df, "cbramod_twophase_pre", "split_fm_lco", cond="unseen"))
    if t:
        res["S2_cbramod_dueFasi_nonviste_vs_caso"] = t
    t = _vs_chance(subj(df, "reve_base", "split_fm_lco", cond="unseen"))
    if t:
        res["S3_reve_nonviste_vs_caso"] = t
    t = _pair(subj(df, "reve_base"), subj(df, "reve_base", variant="bp4-30"), alternative="greater")
    if t:
        res["S4_reve_fuori_4_30Hz"] = t
    ps = {k: v["p"] for k, v in res.items() if "p" in v}
    for k, p in holm(ps).items():
        res[k]["p_holm"] = p
    return res


def ablation_a4_decision(df) -> dict:
    """Regola A.4: l'ablazione della testa va lanciata solo se cbramod_twophase_pre sulle clip non viste è
    significativamente sopra il caso (Wilcoxon unilaterale per soggetto contro 1/9, p < 0.05)."""
    s = subj(df, "cbramod_twophase_pre", "split_fm_lco", cond="unseen")
    if s is None:
        return {"stato": "leave-clip-out a due fasi non ancora disponibile"}
    t = _vs_chance(s)
    return {"ba_unseen": float(s.mean()), "p_unilaterale": t["p"],
            "decisione": "LANCIARE la coda F_bis (ablazione A.4)" if t["p"] < 0.05 else
            "NON lanciare F_bis: risultato robusto alla ricetta (al caso anche con LP-FT)"}


def model_summary(df, configs) -> list[dict]:
    """Tabella di riepilogo: split_fm, T_k, viste, non viste (p contro il caso), memorizzazione."""
    rows = []
    for c in configs:
        r = {"config": c}
        s = subj(df, c)
        if s is not None:
            r["split_fm"] = float(s.mean())
        tks = [subj(df, c, cond=f"tk{k}") for k in range(3)]
        if all(t is not None for t in tks):
            r["T_k"] = float(np.mean([t.mean() for t in tks]))
        for cond in ("seen", "unseen"):
            s = subj(df, c, "split_fm_lco", cond=cond)
            if s is not None:
                r[cond] = float(s.mean())
                if cond == "unseen":
                    r["p_unseen_vs_caso"] = _vs_chance(s)["p"]
        d = delta_stim(df, c, "split_fm_lco")
        if d and "memorization_mean" in d:
            r["memorizzazione"] = d["memorization_mean"]
        if len(r) > 1:
            rows.append(r)
    return rows


def reve_vs_cbramod(df, cb="cbramod_ft_official", rv="reve_base") -> dict:
    """Confronti appaiati per soggetto REVE contro CBraMod (ricetta ufficiale; LoRA dove manca l'ufficiale)."""
    out = {}
    t = _pair(subj(df, rv), subj(df, cb))
    if t:
        out["split_fm"] = t
    for cond in ("seen", "unseen"):
        t = _pair(subj(df, rv, "split_fm_lco", cond=cond), subj(df, cb, "split_fm_lco", cond=cond))
        if t:
            out[f"lco_{cond}"] = t
    for name, (a, b) in {"memorizzazione": (rv, "cbramod_lora_r2")}.items():   # permutati solo per LoRA
        da, db = delta_stim(df, a, "split_fm_lco"), delta_stim(df, b, "split_fm_lco")
        if da and db and "memorization" in da and "memorization" in db:
            sa = np.array(da["memorization"], float)
            sb = np.array(db["memorization"], float)
            out[f"{name} (REVE vs CBraMod LoRA)"] = paired_test(sa, sb)
    dr, dc = stim_decomposition(df, rv, "split_fm"), stim_decomposition(df, "cbramod_lora_r2", "split_fm")
    for k in ("data_quantity", "dstim_size", "dstim_size_corr"):
        if dr and dc and k in dr and k in dc:
            out[f"scomposizione {k} (REVE vs CBraMod LoRA)"] = paired_test(
                np.array(dr[k]["per_subject"], float), np.array(dc[k]["per_subject"], float))
    er, ec = delta_emg(df, rv), delta_emg(df, "cbramod_lora_r2")
    for k in ("bp4_30", "bp4_47", "bp0_30", "noedge", "ocular4", "temporal4", "rand8", "rand4",
              "bp4_30_noedge", "bp4_30_rand8"):
        if er.get(k) is not None and ec.get(k) is not None and er.get("full") is not None and ec.get("full") is not None:
            cr = np.array(er["full"], float) - np.array(er[k], float)
            cc = np.array(ec["full"], float) - np.array(ec[k], float)
            out[f"calo {k} (REVE vs CBraMod LoRA)"] = paired_test(cr, cc)
    return out


def fmt(v: dict) -> str:
    return (f"diff {v['mean_diff']:+.4f} (mediana {v['median_diff']:+.4f}), IC95 "
            f"{np.round(v.get('ci95', [np.nan, np.nan]), 4).tolist()}, peggiorati {v.get('frac_worse', np.nan):.2f}, "
            f"p {v.get('p', np.nan):.3g} ({v.get('test')})" + (f", p Holm {v['p_holm']:.3g}" if "p_holm" in v else "")
            + f", n {v['n']}")


def markdown(df, jobs, gate_r: dict | None = None, timing: dict | None = None) -> list[str]:
    md = ["\n## Sessione 02\n"]
    if gate_r:
        md.append(f"**Gate R (REVE)**: {gate_r}\n")
    md.append("### Seconda famiglia di analisi (Holm su 4, aggiunta successiva)\n")
    for k, v in second_family(df, jobs).items():
        md.append(f"- **{k}**: {fmt(v)}" + (f" — {v['coppia']}" if "coppia" in v else ""))
    md.append(f"\n**Regola A.4**: {ablation_a4_decision(df)}\n")
    md.append("### Riepilogo dei modelli (BA per finestra, media per soggetto)\n")
    md.append("| config | split FM | T_k | viste | non viste | p non viste > caso | memorizzazione |\n|---|---|---|---|---|---|---|")
    for r in model_summary(df, ["cbramod_ft_official", "cbramod_lora_r2", "cbramod_twophase_pre", "cbramod_twophase_scratch",
                                "cbramod_scratch_ft_warmup", "cbramod_ft_official_warmup", "reve_base", "reve_base_scratch",
                                "reve_base_lora16"]):
        f = lambda k: f"{r[k]:.3f}" if k in r else "—"
        md.append(f"| {r['config']} | {f('split_fm')} | {f('T_k')} | {f('seen')} | {f('unseen')} | "
                  f"{('%.3g' % r['p_unseen_vs_caso']) if 'p_unseen_vs_caso' in r else '—'} | {f('memorizzazione')} |")
    md.append("\n### Pretraining sulle clip viste e non viste (Parte B)\n")
    for name, r in pretraining_on_unseen(df, jobs).items():
        md.append(f"- {name} ({r['coppia']}):")
        for k, v in r.items():
            if k != "coppia":
                md.append(f"    - {k}: {fmt(v)}")
    rc = reve_vs_cbramod(df)
    if rc:
        md.append("\n### REVE contro CBraMod (appaiato per soggetto; positivo = REVE più alto / calo maggiore)\n")
        for k, v in rc.items():
            md.append(f"- {k}: {fmt(v)}")
    if timing:
        md.append("\n### Tempi per job e per fase\n")
        md.append("| config | job | min/job | min per fase | epoche per fase | epoca migliore |\n|---|---|---|---|---|---|")
        for c, t in sorted(timing.items()):
            md.append(f"| {c} | {t['n']} | {t['min_job']} | {t['min_fase']} | {t['epoche']} | {t['epoca_migliore']} |")
    return md
