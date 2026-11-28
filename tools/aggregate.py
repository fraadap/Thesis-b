"""Aggrega i risultati scaricati da Kaggle e produce tabelle (markdown + CSV), test statistici e grafici.

    python tools/aggregate.py results_account1_nbA.zip results_account2_nbB.zip --out analysis/
    python tools/aggregate.py incollato.json --out analysis/     # blocco JSON copiato dalla chat

Input accettati: zip `results_*.zip`, file `rows_*.jsonl`, cartelle `pianob_state`, oppure il JSON compatto
(PIANOB_JSON_BEGIN ... END; i punteggi sono già mediati su seed e fold per soggetto).

Famiglia confermativa (Holm, pre-registrata):
  H1 il pretraining aiuta:        cbramod_ft_official vs cbramod_scratch_ft (split_fm)
  H2 l'allineamento aiuta:        tsception_ea vs tsception_noalign (split_fm)
  H3 Δstim^size corretto > 0:     modello migliore (default tsception_ea, --best per cambiarlo): BA_size - BA_unseen
                                  sugli stessi esempi T_k, a parità di quantità di dati, corretto per permutazione
                                  a livello di clip (eval/shortcuts.stim_decomposition). L'indice appaiato
                                  (viste vs non viste, stesso modello) resta come stima di robustezza.
  H4 ΔEMG > controllo casuale:    BA(4-30 Hz, senza 8 canali casuali) - BA(4-30 Hz, senza bordo) > 0
Tutto il resto è ESPLORATIVO.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pianob.eval.shortcuts import (decomposition_lines, delta_emg, delta_stim, stim_decomposition,  # noqa: E402
                                   tk_rows_from_preds)
from pianob.eval.stats import holm, mde_paired, nadeau_bengio, paired_test  # noqa: E402
from pianob.report import (attach_job_times, gate1, latest_valid_jobs, main_view, markdown_table,  # noqa: E402
                           summarize)


def _rows_from_lines(lines):
    out = []
    for l in lines:
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:   # riga troncata (disco pieno a metà scrittura)
                pass
    return out


def _compact_to_rows(cj: dict) -> list[dict]:
    rows = []
    for g in cj.get("groups", []):
        config, protocol, task, variant, cond = g["k"]
        for s, bw, bt in zip(g["subj"], g["bw"], g["bt"]):
            rows.append({"config": config, "protocol": protocol, "task": task, "variant": variant,
                         "test_condition": cond, "subject": s, "ba_win": bw, "ba_trial": bt, "seed": -1, "fold": -1,
                         "kappa": np.nan, "f1w": np.nan})
    return rows


def _recipe_choice_from(paths) -> dict | None:
    """Scelta della ricetta di REVE (diagnosi del Gate R2) negli input: serve prima di filtrare righe e job."""
    for p in map(Path, paths):
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if n.endswith("hpo_best/reve_recipe.json"):
                        return json.loads(z.read(n))
        elif p.is_dir():
            for f in p.rglob("hpo_best/reve_recipe.json"):
                return json.loads(f.read_text())
    return None


def load_inputs(paths) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    from pianob.experiments.queue import set_recipe_choice
    set_recipe_choice(_recipe_choice_from(paths))
    rows, jobs, compact, pred_rows = [], [], [], []
    for p in map(Path, paths):
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if re.search(r"results/rows_.*\.jsonl$", n):
                        rows += _rows_from_lines(io.TextIOWrapper(z.open(n)).read().splitlines())
                    elif re.search(r"results/jobs_.*\.jsonl$", n):
                        dn = n.replace("results/jobs_", "done/done_")
                        done = _rows_from_lines(z.read(dn).decode().splitlines()) if dn in z.namelist() else []
                        jobs += attach_job_times(_rows_from_lines(io.TextIOWrapper(z.open(n)).read().splitlines()),
                                                 done)
                    elif re.search(r"artifacts/preds/.*\.npz$", n):
                        with np.load(io.BytesIO(z.read(n))) as npz:
                            pred_rows += tk_rows_from_preds(Path(n).stem, npz)
        elif p.is_dir():
            for f in p.rglob("rows_*.jsonl"):
                rows += _rows_from_lines(f.read_text().splitlines())
            for f in p.rglob("jobs_*.jsonl"):
                dn = f.parent.parent / "done" / f.name.replace("jobs_", "done_")
                done = _rows_from_lines(dn.read_text().splitlines()) if dn.exists() else []
                jobs += attach_job_times(_rows_from_lines(f.read_text().splitlines()), done)
            for f in p.rglob("artifacts/preds/*.npz"):
                with np.load(f) as npz:
                    pred_rows += tk_rows_from_preds(f.stem, npz)
        elif p.suffix == ".jsonl":
            target = jobs if p.name.startswith("jobs_") else rows
            target += _rows_from_lines(p.read_text().splitlines())
        else:
            txt = p.read_text()
            m = re.search(r"PIANOB_JSON_BEGIN\s*(\{.*\})\s*PIANOB_JSON_END", txt, re.S)
            cj = json.loads(m.group(1) if m else txt)
            compact.append(cj)
            rows += _compact_to_rows(cj)
    df = pd.DataFrame(rows + pred_rows)   # le righe salvate dai job hanno la precedenza (keep="first" sotto)
    if not df.empty:
        from pianob.experiments.queue import invalid_row_mask
        df = df[~invalid_row_mask(df)]
        if "t" in df.columns:   # a parità di chiave vince la riga più recente
            df = df.iloc[::-1].sort_values("t", ascending=False, na_position="last", kind="stable")
        df = df.drop_duplicates(keep="first", subset=["config", "protocol", "fold", "seed", "variant",
                                                      "test_condition", "task", "subject"])
        df = df.drop_duplicates(subset=["config", "protocol", "fold", "seed", "variant", "test_condition", "task",
                                        "subject"], keep="last")
    return df, latest_valid_jobs(jobs), compact


def subj_series(df, config, protocol="split_fm", variant="base", cond="all", task="emo9", metric="ba_win"):
    d = df[(df.config == config) & (df.protocol == protocol) & (df.variant == variant) &
           (df.test_condition == cond) & (df.task == task)]
    return d.groupby("subject")[metric].mean() if not d.empty else None


def paired(df, a, b, **kw):
    sa, sb = subj_series(df, a, **kw), subj_series(df, b, **kw)
    if sa is None or sb is None:
        return None
    idx = sa.index.intersection(sb.index)
    return paired_test(sa[idx].values, sb[idx].values)


def best_by_validation(jobs: list[dict], configs: list[str], protocol: str = "split_fm") -> str | None:
    """Config con la migliore metrica di selezione media in validation (solo job base, task emo9)."""
    sc = {}
    for j in jobs:
        if j.get("config") in configs and f"__{protocol}__" in j["job_id"] and j["job_id"].endswith("__emo9__base") \
                and "val_best" in j:
            sc.setdefault(j["config"], []).append(j["val_best"])
    return max(sc, key=lambda c: np.mean(sc[c])) if sc else None


def auto_best(df) -> str:
    """Modello migliore pre-registrato: BA (finestra, media per soggetto) più alta su split_fm tra quelli che hanno
    sia la scomposizione con permutazioni sia l'ablazione EMG."""
    cands = {}
    for c in df.config.unique():
        dec = stim_decomposition(df, c, "split_fm")
        if dec and "dstim_size_corr" in dec and delta_emg(df, c).get("bp4_30_rand8") is not None:
            cands[c] = subj_series(df, c).mean()
    return max(cands, key=cands.get) if cands else "tsception_ea"


def primary_hypotheses(df, best: str, jobs: list[dict] | None = None) -> dict:
    res = {}
    # H1 (deviazione dichiarata): la ricetta esatta da zero collassa (classe costante, cfr. issue #22) e il
    # warm-up delle LR peggiora il pre-addestrato. Per ciascun modello si usa la ricetta con la migliore
    # metrica di selezione media in VALIDATION (mai il test); senza i record dei job: ufficiale vs warm-up.
    pre = best_by_validation(jobs or [], ["cbramod_ft_official", "cbramod_ft_official_warmup"]) or "cbramod_ft_official"
    scr = best_by_validation(jobs or [], ["cbramod_scratch_ft", "cbramod_scratch_ft_warmup",
                                          "cbramod_scratch_ft_definit_warmup"]) or "cbramod_scratch_ft_warmup"
    r = paired(df, pre, scr)
    if r:
        res["H1_pretraining"] = {**r, "pair": f"{pre} vs {scr} (ricette scelte in validation)"}
    r = paired(df, "tsception_ea", "tsception_noalign")
    if r:
        res["H2_allineamento"] = r
    dec = stim_decomposition(df, best, "split_fm")
    if dec and "dstim_size_corr" in dec:
        c = np.array(dec["dstim_size_corr"]["per_subject"], float)
        c = c[~np.isnan(c)]
        res["H3_delta_stim_size_corr"] = {**paired_test(c, np.zeros_like(c)), "model": best}
    e = delta_emg(df, best)
    if e and "bp4_30_rand8" in e and "bp4_30_noedge" in e:
        res["H4_emg_vs_controllo"] = paired_test(np.array(e["bp4_30_rand8"], float), np.array(e["bp4_30_noedge"], float))
    ps = {k: v["p"] for k, v in res.items() if "p" in v}
    for k, p in holm(ps).items():
        res[k]["p_holm"] = p
    return res


def exploratory(df) -> dict:
    out = {}
    pairs = [("cbramod_ft_official", "cbramod_scratch_ft_warmup"), ("cbramod_ft_official_warmup", "cbramod_ft_official"),
             ("cbramod_ft_official", "cbramod_scratch_ft"), ("eegnet_ea", "eegnet_noalign"), ("cbramod_lora_r2_ea", "cbramod_lora_r2"),
             ("cbramod_lora_r2", "cbramod_ft_official"), ("tsception_ea", "de_asym_lr"),
             ("cbramod_lora_r2", "tsception_ea"), ("cbramod_ft_official", "tsception_ea"),
             ("de_asym_lr_zsub", "de_asym_lr"), ("tsception_ea_bnalpha", "tsception_ea")]
    for a, b in pairs:
        kw = {}
        if a.endswith("_bnalpha"):
            sa = subj_series(df, a, variant="bnalpha_session_n0sel")
            sb = subj_series(df, b)
            if sa is not None and sb is not None:
                idx = sa.index.intersection(sb.index)
                out[f"{a}(session,n0sel) vs {b}"] = paired_test(sa[idx].values, sb[idx].values)
            continue
        r = paired(df, a, b, **kw)
        if r:
            out[f"{a} vs {b}"] = r
    return out


def unseen_lnso10(df) -> dict:
    """Clip nuove con 123 soggetti (lnso10_lco): ogni modello contro il caso (unilaterale) e confronti appaiati;
    ΔEMG di TSception (4-30 Hz senza 8 canali casuali - senza bordo, media delle 2 estrazioni)."""
    out = {}
    models = ["de_asym_lr_zsub", "tsception_ea", "cbramod_lora_r2"]
    ser = {m: subj_series(df, m, protocol="lnso10_lco", cond="unseen") for m in models}
    ser["tsception_ea_bnalpha"] = subj_series(df, "tsception_ea_bnalpha", protocol="lnso10_lco",
                                              variant="bnalpha_session_n0sel", cond="unseen")
    for m, s in ser.items():
        if s is not None:
            out[f"{m} non viste > caso"] = paired_test(s.values, np.full(len(s), 1 / 9), alternative="greater")
    for a, b in [("de_asym_lr_zsub", "tsception_ea"), ("de_asym_lr_zsub", "tsception_ea_bnalpha"),
                 ("de_asym_lr_zsub", "cbramod_lora_r2"), ("tsception_ea", "cbramod_lora_r2")]:
        sa, sb = ser[a], ser[b]
        if sa is not None and sb is not None:
            idx = sa.index.intersection(sb.index)
            out[f"{a} vs {b} (non viste)"] = paired_test(sa[idx].values, sb[idx].values)
    ne = subj_series(df, "tsception_ea", protocol="lnso10", variant="bp4-30_noedge")
    r8 = [subj_series(df, "tsception_ea", protocol="lnso10", variant=f"bp4-30_rand8d{d}") for d in (0, 1)]
    if ne is not None and all(r is not None for r in r8):
        r = sum(r8) / len(r8)
        idx = ne.index.intersection(r.index)
        out["ΔEMG tsception_ea lnso10 (senza 8 casuali - senza bordo, 4-30 Hz)"] = paired_test(
            r[idx].values, ne[idx].values, alternative="greater")
    return out


def power_gate(df, a="tsception_ea", b="de_asym_lr") -> dict:
    """Dopo lnso10: SD delle differenze appaiate per soggetto -> effetto minimo rilevabile (n = 123)."""
    sa, sb = subj_series(df, a, protocol="lnso10"), subj_series(df, b, protocol="lnso10")
    if sa is None or sb is None:
        return {"status": "lnso10 non ancora disponibile"}
    idx = sa.index.intersection(sb.index)
    d = sa[idx] - sb[idx]
    sd = float(d.std(ddof=1))
    mde = mde_paired(sd, len(idx))
    # Nadeau-Bengio sui 10 fold (differenza media per fold)
    folds = {}
    for config in (a, b):
        x = df[(df.config == config) & (df.protocol == "lnso10") & (df.variant == "base") & (df.task == "emo9")]
        folds[config] = x.groupby("fold")["ba_win"].mean()
    nb = nadeau_bengio((folds[a] - folds[b]).dropna().values, n_train=111, n_test=12) if all(len(v) for v in folds.values()) else {}
    return {"configs": [a, b], "n": len(idx), "sd_diff": sd, "mde": mde, "mde_over_2pt": bool(mde > 0.02),
            "nadeau_bengio": nb}


def plots(df, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s = summarize(main_view(df))
    for proto, conds in (("split_fm", ["all"]), ("split_fm_lco", ["seen", "unseen"])):
        x = s[(s.protocol == proto) & (s.variant == "base") & (s.task == "emo9")]
        if x.empty:
            continue
        configs = sorted(x.config.unique(), key=lambda c: x[x.config == c].ba_win.max())
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(configs) + 1.5))
        h = 0.8 / len(conds)
        for i, cond in enumerate(conds):
            y = [x[(x.config == c) & (x.test_condition == cond)].ba_win.mean() for c in configs]
            e = [x[(x.config == c) & (x.test_condition == cond)].ba_win_sd_subj.mean() / np.sqrt(
                max(1, x[(x.config == c) & (x.test_condition == cond)].n_subj.mean())) for c in configs]
            ax.barh(np.arange(len(configs)) + i * h, y, h, xerr=e, label=cond)
        ax.axvline(1 / 9, color="gray", ls="--", lw=1, label="caso (1/9)")
        ax.set_yticks(np.arange(len(configs)) + h * (len(conds) - 1) / 2, configs)
        ax.set_xlabel("BA a livello di finestra (media per soggetto)")
        ax.set_title(proto)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"ba_{proto}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--best", default="auto", help="modello per H3/H4; 'auto' = BA più alta su split_fm tra i "
                                                    "modelli neurali con scomposizione e ablazione EMG")
    ap.add_argument("--sr-boot", type=int, default=2000, help="Sessione 03: ricampionamenti bootstrap (0 = niente IC)")
    ap.add_argument("--sr-perm", type=int, default=5000, help="Sessione 03: permutazioni dei residui entro clip")
    ap.add_argument("--no-h2", action="store_true", help="Sessione 03: salta la rianalisi delle predizioni a 9 classi")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, jobs, compact = load_inputs(a.inputs)
    if df.empty:
        sys.exit("nessun risultato negli input")
    if a.best == "auto":
        a.best = auto_best(df)
        print(f"[H3/H4] modello migliore: {a.best}")
    s = summarize(main_view(df))
    s.to_csv(out / "tabella_principale.csv", index=False)
    md = ["# Risultati Piano B (FACED)\n", markdown_table(s, 1000), ""]
    g1 = gate1(jobs) if jobs else next((c.get("gate1") for c in compact if c.get("gate1")), {})
    md.append(f"**Gate 1**: {g1}\n")
    md.append("## Ipotesi primarie (Holm sulla famiglia di 4)\n")
    prim = primary_hypotheses(df, a.best, jobs)
    for k, v in prim.items():
        md.append(f"- **{k}**: diff media {v['mean_diff']:.4f}, mediana {v['median_diff']:.4f}, dz {v.get('dz', np.nan):.2f}, "
                  f"IC95 {np.round(v.get('ci95', [np.nan, np.nan]), 4).tolist()}, peggiorati {v.get('frac_worse', np.nan):.2f}, "
                  f"p {v.get('p', np.nan):.4g} ({v.get('test')}), p Holm {v.get('p_holm', np.nan):.4g}, n {v['n']}")
    md.append("\n## Indici di scorciatoia\n")
    tasks = sorted(df.task.unique())
    for c, p, t in [(c, p, t) for c in sorted(df.config.unique()) for t in tasks for p in ("split_fm_lco", "lnso10_lco")]:
        d = delta_stim(df, c, p, task=t)
        if d:
            md.append(f"- Δstim {c} [{p}, {t}]: viste {d['seen_mean']:.3f}, non viste {d['unseen_mean']:.3f}, grezzo {d['raw_mean']:.4f}" +
                      (f"; permutate viste {d['seen_perm_mean']:.3f} / non viste {d['unseen_perm_mean']:.3f}; "
                       f"memorizzazione {d['memorization_mean']:.4f}; corretto (vs caso) {d['corr_chance_mean']:.4f}; "
                       f"corretto (prompt) {d['corr_mean']:.4f}" if "corr_mean" in d else ""))
    # varianti (es. bande) che hanno anche modelli permutati sul leave-clip-out
    lco = df[df.protocol == "split_fm_lco"]
    for c in sorted(lco.config.unique()):
        vs = set(lco[lco.config == c].variant)
        for v in sorted(x for x in vs if x != "base" and "perm" not in x and any(y.startswith(x + "_perm") for y in vs)):
            d = delta_stim(df, c, "split_fm_lco", base_variant=v)
            if d and "corr_mean" in d:
                md.append(f"- Δstim {c} [split_fm_lco, {v}]: viste {d['seen_mean']:.3f}, non viste {d['unseen_mean']:.3f}; "
                          f"permutate viste {d['seen_perm_mean']:.3f} / non viste {d['unseen_perm_mean']:.3f}; "
                          f"memorizzazione {d['memorization_mean']:.4f}; corretto (vs caso) {d['corr_chance_mean']:.4f}")
    for c in sorted(df.config.unique()):
        e = delta_emg(df, c)
        if e and len(e) > 2:
            md.append(f"- ΔEMG {c}: " + ", ".join(f"{k[6:]} {e[k]:.4f}" for k in e if k.startswith("delta_")))
    md.append("\n## Quantità di dati contro novità dello stimolo\n")
    for c in sorted(df.config.unique()):
        for t in tasks:
            for bp in ("split_fm", "lnso10"):
                md += decomposition_lines(stim_decomposition(df, c, bp, task=t), f"{c} [{bp}, {t}]")
    md.append("\n## Confronti esplorativi (non corretti)\n")
    for k, v in exploratory(df).items():
        md.append(f"- {k}: diff {v['mean_diff']:.4f}, dz {v.get('dz', np.nan):.2f}, p {v.get('p', np.nan):.4g} ({v.get('test')}), n {v['n']}")
    un = unseen_lnso10(df)
    if un:
        md.append("\n## Clip nuove e EMG con 123 soggetti (lnso10, esplorativo)\n")
        for k, v in un.items():
            md.append(f"- {k}: diff {v['mean_diff']:+.4f}, IC95 {np.round(v['ci95'], 4).tolist()}, "
                      f"peggiorati {v['frac_worse']:.2f}, p {v['p']:.3g} ({v['test']}), n {v['n']}")
    md.append(f"\n## Gate di potenza (lnso10)\n\n{power_gate(df)}\n")
    probes = {j["job_id"]: j["clip_probe"] for j in jobs if "clip_probe" in j}
    for c in compact:
        for k, v in c.get("job_extra", {}).items():
            if "clip_probe" in v:
                probes[k] = v["clip_probe"]
    if probes:
        md.append("## Sonda dell'identità della clip\n")
        for k, v in probes.items():
            md.append(f"- {k}: accuratezza clip {v['clip_acc']:.3f} (caso {v['clip_chance']:.3f}), "
                      f"BA da clip->categoria {v['ba_from_clip']:.3f}")
    tm = {j["job_id"]: j["clip_probe_token_mean"] for j in jobs if "clip_probe_token_mean" in j}
    for k, v in tm.items():
        md.append(f"- {k} (media dei token): accuratezza clip {v['clip_acc']:.3f} (caso {v['clip_chance']:.3f}), "
                  f"BA da clip->categoria {v['ba_from_clip']:.3f}")
    # Sessione 02: seconda famiglia, pretraining su clip non viste, REVE contro CBraMod, Gate R, tempi
    from pianob.eval import session02
    from pianob.report import gate_r, timing
    gr = gate_r(jobs) if jobs else next((c.get("gate_r") for c in compact if c.get("gate_r")), None)
    tmg = timing(jobs) if jobs else next((c.get("timing") for c in compact if c.get("timing")), None)
    md += session02.markdown(df, jobs, gr, tmg)
    # Sessione 03: label da self-report (servono la tabella dei voti e le predizioni negli input)
    from pianob.selfreport import analysis as SA
    sr = SA.run_all([p for p in a.inputs if Path(p).suffix == ".zip" or Path(p).is_dir()], out / "selfreport",
                    rows=df, n_boot=a.sr_boot, n_perm=a.sr_perm, h2=not a.no_h2, log=lambda *x: None)
    if sr:
        md += sr["markdown"]
    second = session02.second_family(df, jobs)
    (out / "risultati.md").write_text("\n".join(md) + "\n")
    json.dump({"primary": prim, "second_family": second}, open(out / "ipotesi_primarie.json", "w"), indent=1, default=str)
    plots(df, out)
    print("\n".join(md))
    print(f"\n-> {out}/risultati.md, tabella_principale.csv, grafici PNG")


if __name__ == "__main__":
    main()
