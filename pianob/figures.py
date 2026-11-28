"""Figure della tesi (testi in inglese), generate dai risultati di tutte le sessioni.

    python tools/make_figures.py <zip o cartelle pianob_state> ... --out figures/

Ogni figura si salva in PDF (vettoriale) e PNG, con un nome indicativo; `INDEX.md` riporta per ognuna una
descrizione e i numeri principali (in italiano, per lo stato del progetto). Le figure che dipendono da run
non ancora disponibili (protocollo entro soggetto, identità della clip, lunghezza della finestra, voti) si
saltano finché i dati non ci sono.

Convenzioni: BA = balanced accuracy per finestra, media per soggetto (media su seed e fold); intervalli al
95% sui soggetti; linea tratteggiata = caso. Colori fissi per famiglia di modello (stessa entità, stesso
colore in tutte le figure); scala divergente blu-rosso con il grigio al caso.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import faced_meta as FM

CHANCE9 = 1 / 9
# palette di riferimento (validata per daltonismo sulle coppie adiacenti); slot fissi per entità
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
TEXT, TEXT2, GRID, NEUTRAL = "#191919", "#666666", "#e6e6e6", "#efefef"

MODELS = {   # config -> (etichetta, colore)
    "de_asym_lr": ("DE", BLUE),
    "de_asym_lr_zsub": ("DE + z-score", BLUE),
    "tsception_noalign": ("TSception", ORANGE),
    "tsception_ea": ("TSception + EA", ORANGE),
    "tsception_ding_ea": ("TSception + EA (10 s)", ORANGE),
    "eegnet_lawhern": ("EEGNet", AQUA),
    "eegnet_lawhern_ea": ("EEGNet + EA", AQUA),
    "cbramod_ft_official": ("CBraMod", VIOLET),
    "cbramod_lora_r2": ("CBraMod (LoRA)", VIOLET),
    "cbramod_scratch_ft_warmup": ("CBraMod, no pretraining", VIOLET),
    "reve_base": ("REVE", MAGENTA),
}
FAMILY = {"features": ("Spectral features", BLUE), "compact": ("Compact networks", ORANGE),
          "foundation": ("Foundation models", VIOLET)}


def family(cfg: str) -> str:
    return "features" if cfg.startswith("de_") else "foundation" if cfg.startswith(("cbramod", "reve")) else "compact"


def label(cfg: str) -> str:
    return MODELS.get(cfg, (cfg, TEXT2))[0]


def color(cfg: str) -> str:
    return MODELS.get(cfg, (cfg, TEXT2))[1]


CLASS_NAMES = [c.capitalize() for c in FM.CATEGORIES]


# ------------------------------------------------------------------ stile
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "text.color": TEXT, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42, "lines.linewidth": 2,
    })
    return plt


def _save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        files.append(p.name)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return files


def _chance(ax, value=CHANCE9, axis="x", text="Chance"):
    (ax.axvline if axis == "x" else ax.axhline)(value, color=TEXT2, ls="--", lw=1, zorder=1)
    if text:
        if axis == "x":
            ax.text(value, 0.01, f" {text}", transform=ax.get_xaxis_transform(), color=TEXT2, fontsize=8,
                    va="bottom", ha="left")
        else:
            ax.text(0.01, value, f"{text}", transform=ax.get_yaxis_transform(), color=TEXT2, fontsize=8,
                    va="bottom", ha="left")


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_red", ["#0e3d7e", "#338ced", "#9dc8f7", NEUTRAL,
                                                          "#f6b3b3", RED, "#8f1d1e"])


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cce3fd", "#84b9f3", "#338ced", "#1757ac",
                                                       "#0e3d7e"])


# ------------------------------------------------------------------ dati
def subj(df, config, protocol="split_fm", cond="all", variant="base", task="emo9", metric="ba_win"):
    d = df[(df.config == config) & (df.protocol == protocol) & (df.test_condition == cond)
           & (df.variant.astype(str) == variant) & (df.task == task)]
    return d.groupby("subject")[metric].mean() if not d.empty else None


def mci(s) -> tuple[float, float]:
    """Media e semiampiezza dell'intervallo al 95% (t di Student) sui soggetti."""
    from scipy.stats import t
    s = np.asarray(s, float)
    s = s[~np.isnan(s)]
    if len(s) < 2:
        return float(np.mean(s)) if len(s) else float("nan"), 0.0
    return float(s.mean()), float(t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s)))


def paired(a, b) -> dict | None:
    if a is None or b is None:
        return None
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        return None
    from .eval.stats import paired_test
    return paired_test(a[idx].values, b[idx].values)


def iter_preds(paths, valid_ids: set | None = None) -> dict:
    """Predizioni per finestra salvate dai job (artifacts/preds/<job>.npz). Per job duplicati in più input
    vale il file più recente. `valid_ids`: solo i job con un record valido (non invalidato)."""
    best = {}
    for p in map(Path, paths):
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                for i in z.infolist():
                    if re.search(r"artifacts/preds/[^/]+\.npz$", i.filename):
                        jid = Path(i.filename).stem
                        if (valid_ids is None or jid in valid_ids) and (jid not in best or i.date_time > best[jid][0]):
                            best[jid] = (i.date_time, ("zip", str(p), i.filename))
        elif p.is_dir():
            for f in p.rglob("artifacts/preds/*.npz"):
                jid = f.stem
                t = tuple(__import__("time").localtime(f.stat().st_mtime)[:6])
                if (valid_ids is None or jid in valid_ids) and (jid not in best or t > best[jid][0]):
                    best[jid] = (t, ("file", str(f), None))
    return {jid: src for jid, (_, src) in best.items()}


def load_pred(src) -> dict:
    kind, path, member = src
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            d = np.load(io.BytesIO(z.read(member)))
            return {k: d[k] for k in d.files}
    d = np.load(path)
    return {k: d[k] for k in d.files}


def window_index(subject, clip) -> np.ndarray:
    """Indice della finestra nel trial: le finestre di test sono salvate per trial e poi nel tempo."""
    key = np.asarray(subject).astype(np.int64) * 1000 + np.asarray(clip).astype(np.int64)
    out = np.zeros(len(key), dtype=int)
    seen: dict = {}
    for i, k in enumerate(key):
        out[i] = seen.get(k, 0)
        seen[k] = out[i] + 1
    return out


# ------------------------------------------------------------------ F1: clip viste contro clip nuove
F1_SETS = [
    ("23 test subjects (80 for training)", "split_fm_lco",
     ["de_asym_lr_zsub", "tsception_ea", "eegnet_lawhern_ea", "cbramod_ft_official", "reve_base"]),
    ("All 123 subjects (10 subject folds)", "lnso10_lco", ["de_asym_lr_zsub", "tsception_ea", "cbramod_lora_r2"]),
]


def fig_seen_vs_new(df, out):
    plt = _plt()
    sets = [(t, p, [c for c in cs if subj(df, c, p, "unseen") is not None]) for t, p, cs in F1_SETS]
    sets = [s for s in sets if s[2]]
    if not sets:
        return None
    fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 0.55 * max(len(s[2]) for s in sets) + 1.6),
                             squeeze=False)
    nums = {}
    for ax, (title, proto, cfgs) in zip(axes[0], sets):
        for i, c in enumerate(cfgs[::-1]):
            ms, hs = mci(subj(df, c, proto, "seen"))
            mu, hu = mci(subj(df, c, proto, "unseen"))
            ax.plot([mu, ms], [i, i], color=GRID, lw=3, zorder=2, solid_capstyle="round")
            ax.errorbar(ms, i, xerr=hs, fmt="o", color=color(c), ms=8, capsize=0, zorder=3)
            ax.errorbar(mu, i, xerr=hu, fmt="o", mfc="white", mec=color(c), mew=2, color=color(c), ms=8,
                        capsize=0, zorder=3)
            ax.text(ms + hs + 0.012, i, f"{ms:.2f}", va="center", fontsize=8, color=TEXT2)
            ax.text(mu - hu - 0.012, i, f"{mu:.2f}", va="center", ha="right", fontsize=8, color=TEXT2)
            nums[f"{label(c)} ({proto})"] = {"seen": round(ms, 3), "new": round(mu, 3)}
        ax.set_yticks(range(len(cfgs)), [label(c) for c in cfgs[::-1]])
        ax.set_xlim(0.0, max(0.8, ax.get_xlim()[1]))
        ax.grid(axis="y", visible=False)
        _chance(ax)
        ax.set_xlabel("Balanced accuracy")
        ax.set_title(title, loc="left")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", color=TEXT2, ms=7),
                Line2D([], [], marker="o", ls="", mfc="white", mec=TEXT2, mew=2, ms=7)],
               ["Clips seen in training", "New clips"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    files = _save(fig, out, "fig01_seen_vs_new_clips")
    return {"id": "F1", "files": files, "title": "Clip viste contro clip nuove per modello",
            "desc": "Per ogni modello: BA sulle clip viste in training (pieno) e sulle clip nuove (vuoto) nel "
                    "leave-clip-out; a sinistra split FM (23 soggetti di test), a destra 10 fold sui 123 "
                    "soggetti. Mostra il rovesciamento della classifica: i modelli più accurati sulle clip viste "
                    "sono i peggiori sulle nuove.", "numbers": nums}


# ------------------------------------------------------------------ F2: clip condivise contro clip nuove
F2_CFGS = ["de_asym_lr", "de_asym_lr_zsub", "tsception_noalign", "tsception_ea", "tsception_ding_ea",
           "eegnet_lawhern", "eegnet_lawhern_ea", "cbramod_ft_official", "cbramod_lora_r2", "reve_base"]


F2_OFFSETS = {"tsception_ding_ea": (6, -10), "de_asym_lr_zsub": (6, 6), "cbramod_ft_official": (6, -12),
              "cbramod_lora_r2": (6, 6), "eegnet_lawhern_ea": (-6, 6)}


def fig_shared_vs_new(df, out):
    plt = _plt()
    pts = []
    for c in F2_CFGS:
        a, b = subj(df, c, "split_fm", "all"), subj(df, c, "split_fm_lco", "unseen")
        if a is not None and b is not None:
            pts.append((c, *mci(a), *mci(b)))
    if len(pts) < 3:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for c, x, hx, y, hy in pts:
        col = FAMILY[family(c)][1]
        ax.errorbar(x, y, xerr=hx, yerr=hy, fmt="o", color=col, ms=8, mec="white", mew=1.5, elinewidth=1,
                    capsize=0, zorder=3)
        dx, dy = F2_OFFSETS.get(c, (6, 4))
        ax.annotate(label(c), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=TEXT,
                    ha="right" if dx < 0 else "left")
    _chance(ax, axis="y", text="Chance")
    ax.axvline(CHANCE9, color=TEXT2, ls="--", lw=1)
    xs, ys = np.array([p[1] for p in pts]), np.array([p[3] for p in pts])
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax.set_xlabel("Balanced accuracy, clips shared with training")
    ax.set_ylabel("Balanced accuracy, new clips")
    ax.set_title(f"Higher accuracy on shared clips does not transfer to new clips (r = {r:.2f})", loc="left",
                 fontsize=9)
    from matplotlib.lines import Line2D
    fams = sorted({family(p[0]) for p in pts}, key=list(FAMILY).index)
    ax.legend([Line2D([], [], marker="o", ls="", color=FAMILY[f][1], ms=7) for f in fams],
              [FAMILY[f][0] for f in fams], loc="upper right")
    files = _save(fig, out, "fig02_shared_vs_new_clip_accuracy")
    return {"id": "F2", "files": files, "title": "Accuratezza con clip condivise contro accuratezza su clip nuove",
            "desc": "Un punto per modello (23 soggetti di test): asse x BA su split FM (clip condivise), asse y "
                    "BA sulle clip nuove del leave-clip-out; colori per famiglia di modelli. La correlazione "
                    "tra le due misure riassume il rovesciamento.",
            "numbers": {"r": round(r, 3), **{label(p[0]): {"shared": round(p[1], 3), "new": round(p[3], 3)} for p in pts}}}


# ------------------------------------------------------------------ F3: scomposizione dell'accuratezza
F3_CFGS = [("de_asym_lr_zsub", "split_fm_lco"), ("tsception_ea", "split_fm_lco"),
           ("tsception_ding_ea", "split_fm_lco"), ("eegnet_lawhern_ea", "split_fm_lco"),
           ("cbramod_lora_r2", "split_fm_lco"), ("reve_base", "split_fm_lco")]


def fig_decomposition(df, out):
    from .eval.shortcuts import delta_stim
    plt = _plt()
    comps = []
    for c, p in F3_CFGS:
        d = delta_stim(df, c, p)
        if not d or "memorization" not in d:
            continue
        seen = np.array(d["raw"]) * 0 + np.nan
        sub = pd.Index(d["subjects"])
        s_seen, s_un = subj(df, c, p, "seen").reindex(sub), subj(df, c, p, "unseen").reindex(sub)
        memo = np.array(d["memorization"])
        emo = s_seen.values - (memo + CHANCE9)
        gen = s_un.values - CHANCE9
        comps.append((c, mci(memo), mci(emo), mci(gen)))
        del seen
    if not comps:
        return None
    names = ["Clip memorization\n(seen clips, shuffled labels)", "Emotion gain on seen clips\n(true - shuffled labels)",
             "Generalization to new clips\n(above chance)"]
    cols = [YELLOW, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(1.25 * len(comps) + 2.5, 3.8))
    w = 0.26
    x = np.arange(len(comps))
    nums = {}
    for k in range(3):
        m = np.array([cc[1 + k][0] for cc in comps]) * 100
        h = np.array([cc[1 + k][1] for cc in comps]) * 100
        ax.bar(x + (k - 1) * w, m, w * 0.92, yerr=h, color=cols[k], label=names[k], error_kw={"elinewidth": 1,
               "ecolor": TEXT2}, zorder=2)
    for i, cc in enumerate(comps):
        nums[label(cc[0])] = {"memorization": round(cc[1][0] * 100, 1), "emotion_gain_seen": round(cc[2][0] * 100, 1),
                              "new_clips_above_chance": round(cc[3][0] * 100, 1)}
    ax.axhline(0, color=TEXT2, lw=1)
    ax.set_xticks(x, [label(cc[0]) for cc in comps], rotation=20, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Balanced accuracy points above chance")
    ax.set_title("What the accuracy above chance is made of (23 test subjects, leave-clip-out)", loc="left")
    ax.legend(loc="upper left", fontsize=7.5, ncol=3, bbox_to_anchor=(0, -0.28))
    files = _save(fig, out, "fig03_accuracy_decomposition")
    return {"id": "F3", "files": files, "title": "Di cosa è fatta l'accuratezza sopra il caso",
            "desc": "Per modello, in punti di BA sopra il caso: memorizzazione della clip (clip viste con label "
                    "permutate a livello di clip), guadagno emotivo sulle clip viste (label vere meno permutate) "
                    "e generalizzazione alle clip nuove. Leave-clip-out su split FM, modelli con run permutate.",
            "numbers": nums}


# ------------------------------------------------------------------ F4: contenuto lento (passa-alto)
def fig_highpass(df, out):
    plt = _plt()
    cuts = [(0, "base"), (1, "bp1-47"), (2, "bp2-47"), (4, "bp4-47")]
    cfgs = ["cbramod_lora_r2", "eegnet_lawhern_ea", "tsception_ea"]
    lines = {}
    for c in cfgs:
        pts = [(hz, *mci(s)) for hz, v in cuts if (s := subj(df, c, variant=v)) is not None]
        if len(pts) >= 2:
            lines[c] = pts
    if not lines:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    nums = {}
    for c, pts in lines.items():
        x, m, h = zip(*pts)
        ls = "-" if list(x) == [0, 1, 2, 4][:len(x)] else ":"     # punteggiato = soglie intermedie non misurate
        a1.errorbar(x, m, yerr=h, color=color(c), marker="o", ms=7, capsize=0, elinewidth=1, ls=ls, label=label(c))
        nums[label(c)] = {f"highpass_{hz}Hz": round(mm, 3) for hz, mm in zip(x, m)}
    a1.set_xticks([0, 1, 2, 4], ["None", "1 Hz", "2 Hz", "4 Hz"])
    a1.set_xlabel("High-pass cut-off (slow activity removed below it)")
    a1.set_ylabel("Balanced accuracy, shared clips")
    _chance(a1, axis="y")
    a1.set_ylim(0, None)
    a1.set_title("Removing slow activity", loc="left", fontsize=9)
    a1.legend(loc="upper right")
    bands = [("base", "Full\nband"), ("bp0-4", "0-4 Hz\nonly"), ("bp4-47", "4-47 Hz\nonly"), ("bp0-30", "0-30 Hz\nonly")]
    bc = [c for c in cfgs if subj(df, c, variant="bp0-4") is not None or subj(df, c, variant="bp0-30") is not None]
    w = 0.8 / max(1, len(bc))
    for i, c in enumerate(bc):
        for j, (v, _) in enumerate(bands):
            s = subj(df, c, variant=v)
            if s is None:
                continue
            m, h = mci(s)
            a2.bar(j + (i - (len(bc) - 1) / 2) * w, m, w * 0.9, yerr=h, color=color(c),
                   label=label(c) if j == 0 else None, error_kw={"elinewidth": 1, "ecolor": TEXT2}, zorder=2)
            nums.setdefault(label(c), {})[v] = round(m, 3)
    a2.set_xticks(range(len(bands)), [b[1] for b in bands])
    a2.grid(axis="x", visible=False)
    _chance(a2, axis="y", text="")
    a2.set_title("Keeping one band", loc="left", fontsize=9)
    a2.set_ylim(0, a1.get_ylim()[1])
    files = _save(fig, out, "fig04_slow_activity_highpass")
    return {"id": "F4", "files": files, "title": "L'impronta lenta: curva del passa-alto e bande",
            "desc": "A sinistra BA su split FM (clip condivise) al crescere della soglia del passa-alto; a destra "
                    "banda piena, solo 0-4 Hz, solo 4-47 Hz e sotto 30 Hz. L'accuratezza dei modelli accurati sta "
                    "sotto 4 Hz, in gran parte sotto 1 Hz; sopra 30 Hz (EMG) non serve.", "numbers": nums}


# ------------------------------------------------------------------ F5: distribuzione per soggetto
def fig_per_subject(df, out):
    plt = _plt()
    cfgs = [c for c in ("de_asym_lr_zsub", "tsception_ea", "cbramod_lora_r2")
            if subj(df, c, "lnso10_lco", "unseen") is not None]
    if not cfgs:
        return None
    fig, ax = plt.subplots(figsize=(1.6 * len(cfgs) + 1.8, 3.8))
    rng = np.random.default_rng(0)
    nums = {}
    for i, c in enumerate(cfgs):
        s = subj(df, c, "lnso10_lco", "unseen").values
        ax.boxplot(s, positions=[i], widths=0.5, showfliers=False, medianprops={"color": TEXT, "lw": 1.5},
                   boxprops={"color": TEXT2}, whiskerprops={"color": TEXT2}, capprops={"color": TEXT2})
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(s)), s, s=10, color=color(c), alpha=0.75, lw=0, zorder=3)
        frac = float((s > CHANCE9).mean())
        nums[label(c)] = {"mean": round(float(s.mean()), 3), "frac_above_chance": round(frac, 3), "n": len(s)}
    ax.set_xticks(range(len(cfgs)), [f"{label(c)}\n{nums[label(c)]['frac_above_chance']:.0%} of subjects\nabove chance"
                                     for c in cfgs])
    ax.grid(axis="x", visible=False)
    _chance(ax, axis="y")
    ax.set_ylabel("Balanced accuracy, new clips (per subject)")
    ax.set_title("New clips: one point per subject (123 subjects)", loc="left")
    files = _save(fig, out, "fig05_new_clips_per_subject")
    return {"id": "F5", "files": files, "title": "Clip nuove: distribuzione per soggetto",
            "desc": "BA sulle clip nuove per ciascuno dei 123 soggetti (leave-clip-out con 10 fold di soggetti), "
                    "con box plot e quota di soggetti sopra il caso.", "numbers": nums}


# ------------------------------------------------------------------ F6: forest plot delle ipotesi
HYP_LABELS = {
    "H1_pretraining": "H1  Pretraining helps (shared clips)",
    "H2_allineamento": "H2  Alignment helps (TSception)",
    "H3_delta_stim_size_corr": "H3  Seen-clip advantage beyond memorization",
    "H4_emg_vs_controllo": "H4  Edge channels beyond random channels",
    "S1_pretraining_nonviste": "S1  Pretraining helps on new clips",
    "S2_cbramod_dueFasi_nonviste_vs_caso": "S2  CBraMod (LP-FT) above chance, new clips",
    "S3_reve_nonviste_vs_caso": "S3  REVE above chance, new clips",
    "S4_reve_bande": "S4  REVE uses activity outside 4-30 Hz",
}


def fig_hypotheses(df, jobs, out, best="cbramod_lora_r2"):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from aggregate import primary_hypotheses
    from .eval.session02 import second_family
    plt = _plt()
    fam = [("Main family (Holm, 4 tests)", primary_hypotheses(df, best, jobs)),
           ("Second family (Holm, 4 tests)", second_family(df, jobs))]
    rows = [(g, k, v) for g, d in fam for k, v in d.items() if isinstance(v, dict) and "ci95" in v]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.4))
    nums = {}
    for i, (g, k, v) in enumerate(rows[::-1]):
        m, (lo, hi) = v["mean_diff"] * 100, (np.array(v["ci95"]) * 100)
        sig = v.get("p_holm", 1) < 0.05
        col = BLUE if g.startswith("Main") else ORANGE
        ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(m, i, "o", ms=8, color=col if sig else "white", mec=col, mew=2)
        ax.text(1.02, i, f"{m:+.1f}  p(Holm) {v.get('p_holm', float('nan')):.2g}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=TEXT2)
        nums[k] = {"diff_points": round(m, 2), "ci95": [round(lo, 2), round(hi, 2)], "p_holm": v.get("p_holm")}
    ax.set_yticks(range(len(rows)), [HYP_LABELS.get(k, k) for _, k, _ in rows[::-1]])
    ax.axvline(0, color=TEXT2, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect (balanced accuracy points, 95% CI); filled = significant after Holm")
    ax.set_title("Pre-registered hypotheses", loc="left")
    files = _save(fig, out, "fig06_hypotheses_forest_plot")
    return {"id": "F6", "files": files, "title": "Forest plot delle ipotesi (prima e seconda famiglia)",
            "desc": "Effetto di ogni ipotesi pre-registrata in punti di BA con IC al 95%; pieno = significativo "
                    "dopo Holm. Prima famiglia in blu (H1-H4), seconda in arancione (S1-S4).", "numbers": nums}


# ------------------------------------------------------------------ F7: pretraining, viste contro nuove
def fig_pretraining(df, out):
    plt = _plt()
    a, b = "cbramod_ft_official", "cbramod_scratch_ft_warmup"
    vals = {c: [mci(subj(df, c, "split_fm_lco", k)) for k in ("seen", "unseen")] for c in (a, b)
            if subj(df, c, "split_fm_lco", "seen") is not None and subj(df, c, "split_fm_lco", "unseen") is not None}
    if len(vals) < 2:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    styles = {a: ("o", "-", "Pretrained"), b: ("s", "--", "No pretraining")}
    for c, v in vals.items():
        mk, ls, lab = styles[c]
        m, h = zip(*v)
        ax.errorbar([0, 1], m, yerr=h, marker=mk, ls=ls, color=VIOLET, ms=7, capsize=0, elinewidth=1, label=lab,
                    mfc=VIOLET if c == a else "white", mew=2)
    d_seen = paired(subj(df, a, "split_fm_lco", "seen"), subj(df, b, "split_fm_lco", "seen"))
    d_new = paired(subj(df, a, "split_fm_lco", "unseen"), subj(df, b, "split_fm_lco", "unseen"))
    for x, d in ((0, d_seen), (1, d_new)):
        if d:
            y = np.mean([vals[a][x][0], vals[b][x][0]]) + (0 if x == 0 else 0.05)
            ax.text(x + (0.1 if x == 0 else -0.1), y, f"pretrained {d['mean_diff'] * 100:+.1f} pts\np = {d['p']:.2g}",
                    ha="left" if x == 0 else "right", va="center", fontsize=8, color=TEXT2)
    ax.set_xticks([0, 1], ["Clips seen in training", "New clips"])
    ax.set_xlim(-0.4, 1.4)
    ax.grid(axis="x", visible=False)
    _chance(ax, axis="y")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("CBraMod: pretraining helps only on familiar clips", loc="left", fontsize=9)
    ax.legend(loc="upper right")
    files = _save(fig, out, "fig07_pretraining_seen_vs_new")
    return {"id": "F7", "files": files, "title": "Pretraining: interazione clip viste x clip nuove",
            "desc": "CBraMod pre-addestrato (ricetta ufficiale) contro da zero (ricetta con warm-up, scelte in "
                    "validation), leave-clip-out su split FM: differenza appaiata sulle clip viste e sulle nuove.",
            "numbers": {"seen": d_seen and round(d_seen["mean_diff"] * 100, 2),
                        "new": d_new and round(d_new["mean_diff"] * 100, 2),
                        "p_seen": d_seen and d_seen["p"], "p_new": d_new and d_new["p"]}}


# ------------------------------------------------------------------ F8: allineamento per famiglia
F8_PAIRS = [("DE: z-score", "de_asym_lr_zsub", "de_asym_lr"),
            ("TSception: EA", "tsception_ea", "tsception_noalign"),
            ("EEGNet: EA", "eegnet_lawhern_ea", "eegnet_lawhern"),
            ("CBraMod: z-score", "cbramod_lora_r2_zsub", "cbramod_lora_r2"),
            ("CBraMod: EA", "cbramod_lora_r2_ea", "cbramod_lora_r2")]


def fig_alignment(df, out):
    plt = _plt()
    res = [(n, a, paired(subj(df, a), subj(df, b))) for n, a, b in F8_PAIRS]
    res = [r for r in res if r[2]]
    if not res:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(res) + 1.3))
    nums = {}
    for i, (n, a, d) in enumerate(res[::-1]):
        m, (lo, hi) = d["mean_diff"] * 100, np.array(d["ci95"]) * 100
        col = FAMILY[family(a)][1]
        ax.barh(i, m, 0.6, color=col, zorder=2)
        ax.plot([lo, hi], [i, i], color=TEXT2, lw=1, zorder=3)
        ax.text((hi + 0.6) if m >= 0 else (lo - 0.6), i, f"{m:+.1f}", va="center", ha="left" if m >= 0 else "right",
                fontsize=8, color=TEXT2)
        nums[n] = {"diff_points": round(m, 2), "p": d["p"]}
    ax.set_yticks(range(len(res)), [r[0] for r in res[::-1]])
    los = [np.array(r[2]["ci95"])[0] * 100 for r in res] + [0]
    his = [np.array(r[2]["ci95"])[1] * 100 for r in res] + [0]
    ax.set_xlim(min(los) - 6, max(his) + 6)
    ax.axvline(0, color=TEXT2, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in balanced accuracy with per-subject alignment (points, shared clips)")
    ax.set_title("Alignment helps compact models and hurts the foundation model", loc="left", fontsize=9)
    files = _save(fig, out, "fig08_alignment_effect")
    return {"id": "F8", "files": files, "title": "Effetto dell'allineamento per soggetto per famiglia di modelli",
            "desc": "Differenza appaiata (allineato meno non allineato) su split FM, con IC al 95%: EA e z-score "
                    "per soggetto su feature, modelli compatti e CBraMod.", "numbers": nums}


# ------------------------------------------------------------------ F11: heatmap clip x finestra
HEAT_MODELS = [("cbramod_lora_r2", "lnso10_lco"), ("tsception_ea", "lnso10_lco"), ("cbramod_ft_official", "split_fm_lco"),
               ("eegnet_lawhern_ea", "split_fm_lco"), ("tsception_ding_ea", "split_fm_lco"), ("reve_base", "split_fm_lco")]


def clip_order():
    lab = np.asarray(FM.CLIP_LABELS)
    order = [c for k in range(len(FM.CATEGORIES)) for c in np.where(lab == k)[0]]
    idx = FM.clip_index_in_category(FM.CLIP_LABELS)
    names = [f"{CLASS_NAMES[lab[c]]} {idx[c] + 1}" for c in order]
    return order, names


def heat_table(preds: dict, config: str, protocol: str) -> dict | None:
    """% di finestre classificate correttamente per (clip, finestra), media sui soggetti di test (e su seed e
    fold), separatamente per clip viste in training e clip nuove."""
    from .data.splits import make_split
    acc = {"seen": {}, "new": {}}
    n_jobs = 0
    for jid, src in preds.items():
        p = jid.split("__")
        if len(p) != 6 or p[0] != config or p[1] != protocol or p[4] != "emo9" or p[5] != "base":
            continue
        fold = int(p[2][1:])
        unseen = set(make_split(protocol, fold).unseen_clips)
        d = load_pred(src)
        w = window_index(d["subject"], d["clip"])
        ok = (d["probs"].astype(np.float32).argmax(1) == d["label"]).astype(float)
        n_jobs += 1
        for cond, m in (("new", np.isin(d["clip"], list(unseen))), ("seen", ~np.isin(d["clip"], list(unseen)))):
            f = pd.DataFrame({"s": d["subject"][m], "c": d["clip"][m], "w": w[m], "ok": ok[m]})
            for (c, wi), v in f.groupby(["c", "w"]).ok:
                acc[cond].setdefault((int(c), int(wi)), []).append(v.groupby(f.loc[v.index, "s"]).mean())
    if not n_jobs:
        return None
    out = {"n_jobs": n_jobs}
    for cond, dd in acc.items():
        n_w = max(w for _, w in dd) + 1 if dd else 3
        M = np.full((28, n_w), np.nan)
        for (c, wi), lst in dd.items():
            M[c, wi] = pd.concat(lst, axis=1).mean(axis=1).mean() * 100     # media per soggetto, poi sui soggetti
        out[cond] = M
    return out


def fig_heatmap(preds, out, config, protocol):
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm
    t = heat_table(preds, config, protocol)
    if t is None:
        return None
    order, names = clip_order()
    n_w = t["seen"].shape[1]
    vmax = max(np.nanmax(t["seen"]), 40)
    norm = TwoSlopeNorm(vmin=0, vcenter=CHANCE9 * 100, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 7.6), sharey=True)
    for ax, cond, title in ((axes[0], "seen", "Clips seen in training"), (axes[1], "new", "New clips")):
        M = t[cond][order]
        im = ax.imshow(M, cmap=_diverging_cmap(), norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(n_w):
                if not np.isnan(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if (v > 0.75 * vmax or v < 3) else TEXT)
        wl = 30 / n_w                                        # finestre di test contigue sugli ultimi 30 s
        ax.set_xticks(range(n_w), [f"{j * wl:g}-{(j + 1) * wl:g} s" for j in range(n_w)],
                      fontsize=8 if n_w <= 3 else 7, rotation=0 if n_w <= 3 else 45)
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        for k in range(1, len(FM.CATEGORIES)):
            b = sum(np.asarray(FM.CLIP_LABELS) < k) - 0.5
            ax.axhline(b, color="white", lw=1.5)
    axes[0].set_yticks(range(28), names, fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.03)
    cb.set_label("Windows classified correctly (%, mean over subjects)\nwhite = chance (11%)", fontsize=8)
    proto = "123 subjects" if protocol.startswith("lnso10") else "23 test subjects"
    fig.suptitle(f"{label(config)}: correct predictions per clip and {30 / n_w:g}-s window ({proto})", x=0.02, ha="left",
                 fontsize=10)
    name = f"fig11_heatmap_clip_window_{config}"
    files = _save(fig, out, name)
    seen_m, new_m = float(np.nanmean(t["seen"])), float(np.nanmean(t["new"]))
    by_w = {cond: [round(float(x), 1) for x in np.nanmean(t[cond], axis=0)] for cond in ("seen", "new")}
    best_new = [names[order.index(c)] for c in np.argsort(-np.nanmean(t["new"], axis=1))[:3]]
    return {"id": "F11", "files": files, "title": f"Heatmap clip x finestra, {label(config)}",
            "desc": f"Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 "
                    f"clip (righe, per categoria) e delle {n_w} finestre da {30 / n_w:g} s (colonne); a sinistra clip viste in "
                    f"training, a destra clip nuove ({protocol}, {t['n_jobs']} job). Bianco = caso.",
            "numbers": {"seen_mean_pct": round(seen_m, 1), "new_mean_pct": round(new_m, 1), "by_window_pct": by_w,
                        "best_new_clips": best_new}}


# ------------------------------------------------------------------ matrici di confusione
CM_MODELS = [("de_asym_lr_zsub", "lnso10_lco"), ("tsception_ea", "lnso10_lco"), ("cbramod_lora_r2", "lnso10_lco"),
             ("eegnet_lawhern_ea", "split_fm_lco"), ("cbramod_ft_official", "split_fm_lco"), ("reve_base", "split_fm_lco")]


def confusion_sum(jobs, config, protocol, cond):
    M = None
    n = 0
    for j in jobs:
        p = j["job_id"].split("__")
        if len(p) == 6 and p[0] == config and p[1] == protocol and p[4] == "emo9" and p[5] == "base" \
                and f"confusion_{cond}" in j:
            C = np.asarray(j[f"confusion_{cond}"], float)
            M = C if M is None else M + C
            n += 1
    return M, n


def fig_confusion(jobs, out, config, protocol):
    plt = _plt()
    Ms = {c: confusion_sum(jobs, config, protocol, c) for c in ("seen", "unseen")}
    if any(m is None for m, _ in Ms.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    nums = {}
    for ax, (cond, title) in zip(axes, (("seen", "Clips seen in training"), ("unseen", "New clips"))):
        M, n = Ms[cond]
        R = M / M.sum(1, keepdims=True) * 100
        im = ax.imshow(R, cmap=_seq_cmap(), vmin=0, vmax=max(60, np.nanmax(R)))
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                ax.text(j, i, f"{R[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if R[i, j] > 35 else TEXT)
        ax.set_xticks(range(9), CLASS_NAMES, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(9), CLASS_NAMES if cond == "seen" else [""] * 9, fontsize=7.5)
        ax.set_xlabel("Predicted class")
        if cond == "seen":
            ax.set_ylabel("True class")
        ax.set_title(title, fontsize=9)
        ax.grid(False)
        nums[cond] = {"diag_mean_pct": round(float(np.mean(np.diag(R))), 1),
                      "per_class_recall_pct": {CLASS_NAMES[k]: round(float(R[k, k]), 1) for k in range(9)}}
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("% of windows of the true class", fontsize=8)
    proto = "123 subjects" if protocol.startswith("lnso10") else "23 test subjects"
    fig.suptitle(f"{label(config)}: confusion matrices ({proto}, leave-clip-out)", x=0.02, ha="left", fontsize=10)
    files = _save(fig, out, f"fig_confusion_{config}")
    return {"id": "CM", "files": files, "title": f"Matrici di confusione, {label(config)}",
            "desc": f"Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), "
                    f"clip viste a sinistra e clip nuove a destra ({protocol}, somma su fold e seed). La diagonale "
                    f"media è la recall media per classe.", "numbers": nums}


# ------------------------------------------------------------------ figure condizionate a run future
LADDER_CFGS = ["de_asym_lr_zsub", "tsception_ding_ea", "eegnet_lawhern_ea", "cbramod_ft_official"]


def fig_protocol_ladder(df, out):
    plt = _plt()
    cfgs = [c for c in LADDER_CFGS if subj(df, c, "split_fm_within", "all") is not None]
    if not cfgs:
        return None
    steps = [("Same subject,\nsame clip", "split_fm_within", "all"), ("Other subjects,\nshared clips", "split_fm", "all"),
             ("Other subjects,\nnew clips", "split_fm_lco", "unseen")]
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    nums = {}
    for c in cfgs:
        pts = [(k, *mci(s)) for k, (_, p, cond) in enumerate(steps) if (s := subj(df, c, p, cond)) is not None]
        k, m, h = zip(*pts)
        ax.errorbar(k, m, yerr=h, marker="o", color=color(c), ms=7, capsize=0, elinewidth=1, label=label(c))
        nums[label(c)] = {steps[i][1]: round(mm, 3) for i, mm in zip(k, m)}
    ax.set_xticks(range(3), [s[0] for s in steps])
    ax.grid(axis="x", visible=False)
    _chance(ax, axis="y")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("The evaluation protocol decides the accuracy", loc="left")
    ax.legend(loc="upper right")
    files = _save(fig, out, "fig09_protocol_ladder")
    return {"id": "F9", "files": files, "title": "Scala dei protocolli di valutazione",
            "desc": "BA dello stesso modello con tre protocolli: finestre diverse dello stesso soggetto e della "
                    "stessa clip (entro soggetto), altri soggetti con clip condivise (split FM), altri soggetti con "
                    "clip nuove (leave-clip-out). 23 soggetti di test.", "numbers": nums}


def fig_clip_identity(df, out):
    plt = _plt()
    cfgs = [c for c in LADDER_CFGS + ["tsception_ea", "reve_base"] if subj(df, c, task="clip28") is not None]
    if len(cfgs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    nums = {}
    for c in cfgs:
        x, hx = mci(subj(df, c, task="clip28"))
        y, hy = mci(subj(df, c, "split_fm_lco", "seen")) if subj(df, c, "split_fm_lco", "seen") is not None else mci(subj(df, c))
        ax.errorbar(x, y, xerr=hx, yerr=hy, fmt="o", color=color(c), ms=8, mec="white", mew=1.5, elinewidth=1)
        ax.annotate(label(c), (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
        nums[label(c)] = {"clip_identity_ba": round(x, 3), "emotion_seen_ba": round(y, 3)}
    ax.axvline(1 / 28, color=TEXT2, ls="--", lw=1)
    ax.text(1 / 28, 1.0, " Chance (1/28)", transform=ax.get_xaxis_transform(), fontsize=8, color=TEXT2, va="bottom")
    _chance(ax, axis="y", text="Chance (1/9)")
    ax.set_xlabel("Balanced accuracy, which of the 28 clips (other subjects)")
    ax.set_ylabel("Balanced accuracy, emotion on seen clips")
    ax.set_title("Models that recognize the clip score high on emotions", loc="left", fontsize=9)
    files = _save(fig, out, "fig10_clip_identity_vs_emotion")
    return {"id": "F10", "files": files, "title": "Riconoscimento della clip contro riconoscimento dell'emozione",
            "desc": "Asse x: BA nel riconoscere quale delle 28 clip si sta guardando (soggetti di test di split FM); "
                    "asse y: BA sulle emozioni con clip viste. Un punto per modello.", "numbers": nums}


def fig_window_length(df, out):
    plt = _plt()
    groups = {"de_asym_lr_zsub": ["de_asym_lr_zsub_w2", "de_asym_lr_zsub_w4", "de_asym_lr_zsub"],
              "tsception_ding_ea": ["tsception_ding_ea_w2", "tsception_ding_ea_w4", "tsception_ding_ea"],
              "cbramod_ft_official": ["cbramod_ft_official_w4", "cbramod_ft_official"]}
    lens = {"_w2": 2, "_w4": 4}
    have = {g: [(lens.get(c[-3:], 10), c) for c in cs if subj(df, c, "split_fm_lco", "unseen") is not None]
            for g, cs in groups.items()}
    have = {g: v for g, v in have.items() if len(v) >= 2}
    if not have:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    nums = {}
    for ax, cond, title in ((axes[0], "seen", "Clips seen in training"), (axes[1], "unseen", "New clips")):
        for g, v in have.items():
            x, m, h = zip(*[(L, *mci(subj(df, c, "split_fm_lco", cond))) for L, c in sorted(v)])
            ax.errorbar(x, m, yerr=h, marker="o", color=color(g), ms=7, capsize=0, elinewidth=1, label=label(g))
            nums.setdefault(label(g), {})[cond] = {f"{L}s": round(mm, 3) for L, mm in zip(x, m)}
        ax.set_xticks([2, 4, 10], ["2 s", "4 s", "10 s"])
        ax.set_xlabel("Window length")
        ax.set_title(title, fontsize=9)
        _chance(ax, axis="y", text="")
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(loc="upper right")
    fig.suptitle("Longer windows raise accuracy on seen clips, not on new clips", x=0.02, ha="left", fontsize=10)
    files = _save(fig, out, "fig13_window_length")
    return {"id": "F13", "files": files, "title": "Lunghezza della finestra: clip viste e clip nuove",
            "desc": "BA nel leave-clip-out su split FM con finestre da 2, 4 e 10 s, clip viste e clip nuove.",
            "numbers": nums}


def fig_selfreport_variance(paths, out):
    """F12 (Sessione 03): quota di varianza dei voti spiegata dalla clip, dal soggetto e residua, per item."""
    plt = _plt()
    from .selfreport import metrics as SM
    from .selfreport.table import ITEMS
    df = None
    for p in map(Path, paths):
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                n = next((n for n in z.namelist() if n.endswith("selfreport/selfreport.parquet")), None)
                if n:
                    df = pd.read_parquet(io.BytesIO(z.read(n)))
        elif p.is_dir():
            f = next(iter(p.rglob("selfreport/selfreport.parquet")), None)
            if f is not None:
                df = pd.read_parquet(f)
        if df is not None:
            break
    if df is None or df.subject.nunique() < 20:
        return None
    rows = []
    for it in ITEMS:
        Y = df.pivot_table(index="subject", columns="clip", values=it).dropna()
        a = SM.anova2(Y.to_numpy(float))
        rows.append((it, a["share_clip"], a["share_subject"], a["share_residual"], a["icc2_k"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for k, (name, col) in enumerate((("Clip", VIOLET), ("Subject", ORANGE), ("Subject x clip (individual)", BLUE))):
        v = np.array([r[1 + k] for r in rows]) * 100
        ax.bar(x, v, 0.7, bottom=bottom, color=col, label=name, edgecolor="white", lw=1, zorder=2)
        bottom += v
    ax.set_xticks(x, [r[0].capitalize() for r in rows], rotation=35, ha="right")
    ax.set_ylabel("Share of rating variance (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    files = _save(fig, out, "fig12_rating_variance_components")
    return {"id": "F12", "files": files, "title": "Voti: varianza della clip, del soggetto e individuale",
            "desc": "Per ogni item dei voti, quota di varianza attribuita alla clip, al soggetto e all'interazione "
                    "soggetto x clip (componente individuale), da ANOVA a due vie a effetti casuali.",
            "numbers": {r[0]: {"clip": round(r[1] * 100, 1), "subject": round(r[2] * 100, 1),
                               "individual": round(r[3] * 100, 1), "icc2k": round(r[4], 3)} for r in rows}}


# ------------------------------------------------------------------ tutto
def make_all(df, jobs, paths, out: Path, log=print) -> list[dict]:
    out = Path(out)
    valid = {j["job_id"] for j in jobs}
    preds = iter_preds(paths, valid)
    log(f"[figure] {len(df)} righe, {len(jobs)} job, {len(preds)} file di predizioni")
    made = []
    steps = [lambda: fig_seen_vs_new(df, out), lambda: fig_shared_vs_new(df, out), lambda: fig_decomposition(df, out),
             lambda: fig_highpass(df, out), lambda: fig_per_subject(df, out), lambda: fig_hypotheses(df, jobs, out),
             lambda: fig_pretraining(df, out), lambda: fig_alignment(df, out),
             lambda: fig_protocol_ladder(df, out), lambda: fig_clip_identity(df, out),
             lambda: fig_window_length(df, out), lambda: fig_selfreport_variance(paths, out)]
    steps += [lambda c=c, p=p: fig_heatmap(preds, out, c, p) for c, p in HEAT_MODELS]
    steps += [lambda c=c, p=p: fig_confusion(jobs, out, c, p) for c, p in CM_MODELS]
    for f in steps:
        try:
            r = f()
        except Exception as e:      # una figura che fallisce non deve fermare le altre
            import traceback
            log(f"[figure] errore: {e!r}\n{traceback.format_exc()}")
            continue
        if r:
            made.append(r)
            log(f"[figure] {r['id']}: {', '.join(r['files'])}")
    write_index(made, out)
    return made


def write_index(made: list[dict], out: Path) -> None:
    md = ["# Figure della tesi\n", "Generate da `tools/make_figures.py` (testi delle figure in inglese). BA = balanced "
          "accuracy per finestra, media per soggetto; IC al 95% sui soggetti.\n"]
    for r in made:
        md.append(f"## {r['id']} — {r['title']}\n")
        md.append(f"File: {', '.join('`' + f + '`' for f in r['files'])}\n")
        md.append(r["desc"] + "\n")
        md.append("Numeri principali: `" + json.dumps(r["numbers"], ensure_ascii=False, default=str) + "`\n")
    (out / "INDEX.md").write_text("\n".join(md))
    (out / "index.json").write_text(json.dumps(made, indent=1, ensure_ascii=False, default=str))
