"""Passo 0 della Sessione 03: tabella dei voti di FACED (una riga per soggetto e clip).

Fonte: `After_remarks.mat` / `After_remarks.pkl` per soggetto (cartelle `sub000` ... `sub122`), dentro il
Dataset Kaggle dei dati FACED. Struttura attesa (da una guida di terze parti, quindi verificata qui sui file):
`score` 28 x 12 (voti 0-7), `trial` (ordine di presentazione), `vid` (indice canonico della clip, 1-28),
`Accuracy` / `ResponseTime` (compito aritmetico: salvati a parte, non usati).

Ordine delle 12 colonne di `score` (confermato da DataStructureOfBehaviouralData.xlsx, 2026-11-27):
  joy, tenderness, inspiration, amusement, anger, disgust, fear, sadness, arousal, valence, familiarity, liking
Scala 0-7: valence 0 = molto negativo, 7 = molto positivo; gli altri item 0 = per niente, 7 = moltissimo.
`trial` = ordine di presentazione delle 28 clip, `vid` = indice della clip (1-28): il file non dice se le righe di
`score` seguono l'uno o l'altro, quindi l'allineamento resta deciso dai dati (concordanza).
`Accuracy` = accuratezza media del compito aritmetico in ogni intervallo tra i blocchi; `ResponseTime` = tempi dei
20 problemi per intervallo (oltre 4 s = risposta mancata).

Scelte (tutte registrate in `meta_selfreport.json`):
- l'allineamento delle righe di `score` alle clip (ordine di presentazione oppure canonico) si decide dai dati,
  con la concordanza argmax degli 8 item discreti contro la categoria della clip;
- z-score e split alla mediana entro soggetto sulle 28 clip: usano solo i voti del soggetto stesso;
- pareggi alla mediana risolti con rumore seminato (SEED_TIES); `bin_strict` esclude i trial in pareggio;
- l'hash registrato è quello del contenuto canonico della tabella (CSV con formato fisso), non dei byte del
  parquet, che dipendono dalla versione di pyarrow: così i due notebook lo confrontano anche con librerie diverse.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import faced_meta as FM

ITEMS = ["joy", "tenderness", "inspiration", "amusement", "anger", "disgust", "fear", "sadness",
         "arousal", "valence", "familiarity", "liking"]
DISCRETE = ITEMS[:8]
# item discreto -> indice della categoria FACED (0 anger ... 8 tenderness; 4 = neutral non ha item)
ITEM_CATEGORY = {"joy": 7, "tenderness": 8, "inspiration": 6, "amusement": 5,
                 "anger": 0, "disgust": 1, "fear": 2, "sadness": 3}
DERIVED_ITEMS = ["valence", "arousal", "liking", "familiarity"]
SEED_TIES = 20261127
SEED_ARGMAX = 20261128
MIN_ALIGN_DIFF = 0.2          # §1.1: differenza minima di concordanza tra i due allineamenti
MIN_CONCORDANCE = 0.2         # §1.2: sotto questa media (caso 1/8 = 0.125) la mappatura è sbagliata
MIN_COLUMN_MATCHES = 6        # §1.1: item discreti con il massimo sulla propria categoria (su 8)
# Hash del contenuto canonico della tabella registrato dopo la prima esecuzione reale (None = da registrare).
EXPECTED_SELFREPORT_SHA: str | None = None


class SelfReportError(RuntimeError):
    """Controllo bloccante del passo 0 fallito: fermarsi e chiedere all'utente."""


# ------------------------------------------------------------------ lettura dei file
def _simplify(obj):
    """Struct MATLAB / record numpy / oggetti annidati -> dict, list, np.ndarray."""
    if isinstance(obj, dict):
        return {k: _simplify(v) for k, v in obj.items() if not str(k).startswith("__")}
    if hasattr(obj, "_fieldnames"):                                   # scipy mat_struct
        return {f: _simplify(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype.names:               # array strutturato / record
        if obj.size == 1:
            return {n: _simplify(obj[n].item() if obj[n].size == 1 else obj[n]) for n in obj.dtype.names}
        return [{n: _simplify(r[n]) for n in obj.dtype.names} for r in obj.ravel()]
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [_simplify(x) for x in obj.ravel()]
    if isinstance(obj, (list, tuple)):
        return [_simplify(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return {c: obj[c].to_numpy() for c in obj.columns}
    return obj


def _read_h5(path):
    import h5py

    def conv(f, node):
        if isinstance(node, h5py.Group):
            return {k: conv(f, node[k]) for k in node.keys() if not k.startswith("#")}
        arr = node[()]
        if arr.dtype == h5py.ref_dtype:
            return [conv(f, f[r]) for r in arr.ravel()]
        return np.asarray(arr).T          # MATLAB salva in ordine colonna
    with h5py.File(path, "r") as f:
        return {k: conv(f, f[k]) for k in f.keys() if not k.startswith("#")}


def read_any(path) -> dict:
    path = Path(path)
    if path.suffix == ".pkl":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return _simplify(obj) if isinstance(obj, dict) else {"After_remarks": _simplify(obj)}
    try:
        from scipy.io import loadmat
        return _simplify(loadmat(path, squeeze_me=True, struct_as_record=False))
    except NotImplementedError:                                       # v7.3 = HDF5
        return _simplify(_read_h5(path))


def _ci_get(d: dict, name: str):
    for k, v in d.items():
        if str(k).lower() == name.lower():
            return v
    return None


def parse_after_remarks(raw: dict) -> dict:
    """Normalizza il contenuto di After_remarks in score (28, 12), trial (28,), vid (28,) e campi extra."""
    rem = _ci_get(raw, "After_remarks")
    if rem is None:                       # nei .mat di FACED la variabile si chiama After_remark (singolare)
        rem = _ci_get(raw, "After_remark")
    if rem is None:
        keys = [k for k in raw if not str(k).startswith("__")]
        rem = raw[keys[0]] if len(keys) == 1 and not isinstance(raw[keys[0]], np.ndarray) else raw
    if isinstance(rem, list) and rem and isinstance(rem[0], dict):           # struct array 1 x 28
        fields = {k for r in rem for k in r}
        rem = {f: [r.get(f) for r in rem] for f in fields}
    if not isinstance(rem, dict):
        raise SelfReportError(f"After_remarks con struttura inattesa: {type(rem)}")
    score, trial, vid = _ci_get(rem, "score"), _ci_get(rem, "trial"), _ci_get(rem, "vid")
    if score is None or trial is None or vid is None:
        raise SelfReportError(f"After_remarks senza score/trial/vid: campi {sorted(map(str, rem))}")
    score = np.asarray([np.ravel(np.asarray(s, dtype=float)) for s in score]) if isinstance(score, list) \
        else np.asarray(score, dtype=float)
    if score.shape == (12, 28):
        score = score.T
    if score.shape != (28, 12):
        raise SelfReportError(f"score con forma {score.shape}, attesa 28 x 12")
    trial = np.ravel(np.asarray(trial, dtype=float))
    vid = np.ravel(np.asarray(vid, dtype=float))
    if trial.shape != (28,) or vid.shape != (28,):
        raise SelfReportError(f"trial {trial.shape} / vid {vid.shape}: attesi 28 valori")
    extra = {}
    for k, v in rem.items():         # compito aritmetico tra i blocchi: per blocco (e 20 problemi per blocco)
        if str(k).lower() in ("accuracy", "responsetime"):
            try:
                extra[k] = np.ravel(np.asarray(v, dtype=float))
            except (TypeError, ValueError):          # struttura annidata irregolare: appiattita a mano
                extra[k] = np.asarray([float(x) for x in np.ravel(np.asarray(v, dtype=object))
                                       for x in np.ravel(np.asarray(x, dtype=float))])
    return {"score": score, "trial": trial.astype(int), "vid": vid.astype(int), "extra": extra,
            "fields": {str(k): _describe(v) for k, v in rem.items()}}


def _describe(v) -> str:
    """Forma e tipo di un campo (anche per liste irregolari, es. Accuracy vuoto in alcuni trial)."""
    try:
        return f"{np.shape(v)} {type(v).__name__}"
    except ValueError:
        return f"lista di {len(v)} elementi irregolari ({type(v).__name__})"


def find_subject_files(roots=None) -> dict[int, Path]:
    """sub000 ... sub122 -> file After_remarks (il .mat ha la precedenza sul .pkl se ci sono entrambi)."""
    from .. import kaggle_env
    hits = {}
    for name in ("After_remarks.pkl", "After_remarks.mat"):
        for p in kaggle_env.find_all(name, roots=roots, max_depth=9):
            m = re.search(r"sub(\d{3})", str(p.parent), re.I)
            if m:
                hits[int(m.group(1))] = p
    return dict(sorted(hits.items()))


# ------------------------------------------------------------------ allineamento e controlli
def _argmax8(score_rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Categoria dell'item discreto più votato; pareggi tra item risolti a caso (seminato)."""
    d = score_rows[:, :8] + rng.uniform(0, 1e-6, size=(len(score_rows), 8))
    return np.array([ITEM_CATEGORY[DISCRETE[k]] for k in d.argmax(1)])


def concordance(score: np.ndarray, clips: np.ndarray, seed: int = SEED_ARGMAX) -> float:
    """Frazione dei trial non neutri in cui l'argmax degli 8 item è la categoria della clip."""
    cat = np.asarray(FM.CLIP_LABELS)[clips]
    am = _argmax8(score, np.random.default_rng(seed))
    nn = cat != FM.NEUTRAL_CLASS
    return float((am[nn] == cat[nn]).mean())


def choose_alignment(parsed: dict[int, dict]) -> dict:
    """Riga k di score <-> clip vid[k]-1 (righe in ordine di presentazione) oppure <-> clip k (canonico)."""
    conc = {"vid": [], "canonical": []}
    identical = True
    for s, p in parsed.items():
        vid = p["vid"] - 1
        if sorted(vid.tolist()) != list(range(28)):
            raise SelfReportError(f"sub{s:03d}: vid non è una permutazione di 1-28 ({p['vid'].tolist()})")
        identical &= bool((vid == np.arange(28)).all())
        conc["vid"].append(concordance(p["score"], vid))
        conc["canonical"].append(concordance(p["score"], np.arange(28)))
    mv, mc = float(np.mean(conc["vid"])), float(np.mean(conc["canonical"]))
    out = {"concordance_vid": mv, "concordance_canonical": mc, "vid_identity_everywhere": identical}
    if identical:
        out["alignment"] = "vid"           # i due allineamenti coincidono
    elif abs(mv - mc) < MIN_ALIGN_DIFF:
        raise SelfReportError(f"allineamento delle righe indeciso: concordanza con vid {mv:.3f}, canonico "
                              f"{mc:.3f} (differenza < {MIN_ALIGN_DIFF}). Serve una decisione dell'utente.")
    else:
        out["alignment"] = "vid" if mv > mc else "canonical"
    return out


def canonical_rows(p: dict, alignment: str) -> tuple[np.ndarray, np.ndarray]:
    """score (28, 12) e posizione di presentazione (28,) indicizzati per clip canonica 0-27."""
    clips = p["vid"] - 1 if alignment == "vid" else np.arange(28)
    order = np.argsort(clips)
    return p["score"][order], p["trial"][order]


def check_stimuli_info(roots=None) -> dict:
    """Stimuli_info.xlsx: la categoria di ogni vid deve coincidere con la mappa clip -> classe del codice."""
    from .. import kaggle_env
    paths = kaggle_env.find_all("Stimuli_info.xlsx", roots=roots, max_depth=9)
    if not paths:
        return {"status": "file non trovato: verifica affidata alla concordanza (§1.2)"}
    df = pd.read_excel(paths[0])
    cats = {c.lower(): i for i, c in enumerate(FM.CATEGORIES)}
    emo_col = next((c for c in df.columns if df[c].astype(str).str.strip().str.lower().isin(cats).sum() >= 20), None)
    vid_col = next((c for c in df.columns if pd.to_numeric(df[c], errors="coerce").dropna().astype(int)
                    .isin(range(1, 29)).sum() >= 28 and re.search(r"vid|video|clip|no|index|id", str(c), re.I)), None)
    if emo_col is None or vid_col is None:
        raise SelfReportError(f"Stimuli_info.xlsx: colonne non riconosciute {list(df.columns)}\n{df.head(10)}")
    bad = []
    for _, r in df.iterrows():
        v = pd.to_numeric(r[vid_col], errors="coerce")
        e = str(r[emo_col]).strip().lower()
        if pd.isna(v) or e not in cats or not 1 <= int(v) <= 28:
            continue
        if FM.CLIP_LABELS[int(v) - 1] != cats[e]:
            bad.append((int(v), e, FM.CATEGORIES[FM.CLIP_LABELS[int(v) - 1]]))
    if bad:
        raise SelfReportError(f"Stimuli_info.xlsx non coincide con la mappa clip -> classe: {bad}")
    return {"status": "coincide", "file": str(paths[0]), "columns": [str(vid_col), str(emo_col)]}


def describe_structure_file(roots=None) -> str | None:
    from .. import kaggle_env
    paths = kaggle_env.find_all("DataStructureOfBehaviouralData.xlsx", roots=roots, max_depth=9)
    if not paths:
        return None
    try:
        return pd.read_excel(paths[0], header=None).fillna("").to_string(max_rows=60)
    except Exception as e:  # solo descrittivo
        return f"illeggibile: {e!r}"


def column_check(df: pd.DataFrame) -> dict:
    """Media di ogni item per categoria: ogni item discreto deve avere il massimo sulla propria categoria;
    la valenza deve essere più alta sulle clip positive che sulle negative."""
    tab = df.groupby("category")[ITEMS].mean().reindex(FM.CATEGORIES)
    matches = {it: FM.CATEGORIES[int(np.argmax(tab[it].to_numpy()))] == FM.CATEGORIES[ITEM_CATEGORY[it]]
               for it in DISCRETE}
    val = df.assign(v=np.asarray(FM.CLIP_VALENCE)[df["clip"].to_numpy(int)]).groupby("v")["valence"].mean()
    out = {"table": tab.round(2).to_dict(), "item_max_on_own_category": matches,
           "n_matches": int(sum(matches.values())), "valence_pos_minus_neg": float(val.get(1) - val.get(-1))}
    if out["n_matches"] < MIN_COLUMN_MATCHES or out["valence_pos_minus_neg"] <= 0:
        raise SelfReportError("ordine delle colonne di score non confermato: medie item x categoria\n"
                              + tab.round(2).to_string() + f"\nvalenza positive - negative {out['valence_pos_minus_neg']:.2f}")
    return out


# ------------------------------------------------------------------ colonne derivate
def add_derived(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """z, bin, bin_strict per valence/arousal/liking/familiarity (entro soggetto, 28 clip); argmax8; concordant."""
    df = df.sort_values(["subject", "clip"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED_TIES)
    info = {"constant": {}, "ties_at_median": {}, "seed_ties": SEED_TIES}
    for it in DERIVED_ITEMS:
        z, b, bs = (np.full(len(df), np.nan) for _ in range(3))
        consts, ties = [], {}
        for s, idx in df.groupby("subject").indices.items():
            v = df[it].to_numpy(float)[idx]
            if np.nanstd(v) == 0 or np.isnan(v).any():
                consts.append(int(s))
                continue
            z[idx] = (v - v.mean()) / v.std()
            med = np.median(v)
            ties[int(s)] = int((v == med).sum())
            order = np.argsort(v + rng.uniform(0, 1e-6, len(v)), kind="stable")
            bb = np.zeros(len(v))
            bb[order[len(v) // 2:]] = 1                       # 14 sopra, 14 sotto
            b[idx] = bb
            bs[idx] = np.where(v > med, 1.0, np.where(v < med, 0.0, np.nan))
        df[f"{it}_z"], df[f"{it}_bin"], df[f"{it}_bin_strict"] = z, b, bs
        info["constant"][it] = consts
        info["ties_at_median"][it] = ties
    arng = np.random.default_rng(SEED_ARGMAX)
    am = np.empty(len(df), dtype=int)
    for s, idx in df.groupby("subject").indices.items():
        am[idx] = _argmax8(df[DISCRETE].to_numpy(float)[idx], arng)
    df["argmax8"] = am
    df["concordant"] = np.where(df["category_idx"] == FM.NEUTRAL_CLASS, np.nan,
                                (am == df["category_idx"]).astype(float))
    return df, info


# ------------------------------------------------------------------ costruzione
def build_table(parsed: dict[int, dict], log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """parsed: soggetto -> output di parse_after_remarks. Ritorna (tabella, compito aritmetico, meta)."""
    align = choose_alignment(parsed)
    log(f"[voti] allineamento delle righe: {align}")
    rows, task_rows = [], []
    for s, p in parsed.items():
        score, pos = canonical_rows(p, align["alignment"])
        for c in range(28):
            rows.append({"subject": s, "clip": c, "vid": c + 1, "trial_position": int(pos[c]),
                         "category": FM.CATEGORIES[FM.CLIP_LABELS[c]], "category_idx": FM.CLIP_LABELS[c],
                         **{it: float(score[c, k]) for k, it in enumerate(ITEMS)}})
        if p["extra"]:
            task_rows.append({"subject": s, **{k: v.tolist() for k, v in p["extra"].items()}})
    df = pd.DataFrame(rows)
    sc = df[ITEMS].to_numpy()
    qc = {"n_nan": int(np.isnan(sc).sum()), "n_out_of_scale": int(((sc < 0) | (sc > 7)).sum())}
    cols = column_check(df)
    df, der = add_derived(df)
    conc = df[df.category_idx != FM.NEUTRAL_CLASS].groupby("subject")["concordant"].mean()
    if conc.mean() < MIN_CONCORDANCE:
        raise SelfReportError(f"concordanza media {conc.mean():.3f} vicina al caso (1/8): mappatura sbagliata")
    vdiff = df.assign(v=np.asarray(FM.CLIP_VALENCE)[df["clip"].to_numpy(int)]).groupby(["subject", "v"])["valence"].mean().unstack()
    meta = {"alignment": align, "columns": cols, "qc": qc, "derived": der,
            "concordance": {"mean": float(conc.mean()), "median": float(conc.median()),
                            "per_subject": conc.round(3).to_dict(),
                            "flagged_le_1_8": [int(s) for s in conc.index[conc <= 1 / 8]]},
            "valence_pos_gt_neg_fraction": float(((vdiff[1] - vdiff[-1]) > 0).mean()),
            "n_subjects": int(df["subject"].nunique())}
    return df, pd.DataFrame(task_rows), meta


def canonical_hash(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.sort_values(["subject", "clip"]).to_csv(buf, index=False, float_format="%.6g", na_rep="NA")
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def write_outputs(df, task_df, meta, out_dir: Path, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta["sha256"] = canonical_hash(df)
    try:
        df.to_parquet(out_dir / "selfreport.parquet", index=False)
        if len(task_df):
            task_df.to_parquet(out_dir / "selfreport_task.parquet", index=False)
    except ImportError:                        # senza pyarrow: CSV (stesso contenuto, stesso hash)
        df.to_csv(out_dir / "selfreport.csv", index=False)
    (out_dir / "meta_selfreport.json").write_text(json.dumps(meta, indent=1, default=str))
    log(f"[voti] {len(df)} righe, {df["subject"].nunique()} soggetti -> {out_dir} (sha256 {meta['sha256'][:12]}...)")
    return meta


def verify_hash(meta: dict) -> None:
    if EXPECTED_SELFREPORT_SHA and meta["sha256"] != EXPECTED_SELFREPORT_SHA:
        raise SelfReportError(f"hash della tabella dei voti {meta['sha256']} diverso da quello registrato "
                              f"{EXPECTED_SELFREPORT_SHA}: i due notebook non userebbero gli stessi voti")


def selfreport_dir() -> Path:
    from .. import kaggle_env
    return kaggle_env.state_dir() / "selfreport"


def run_step0(roots=None, out_dir: Path | None = None, synthetic: bool = False, log=print) -> dict:
    """Passo 0 completo (comune ai notebook H e I): lettura, controlli bloccanti, tabella, hash."""
    out_dir = out_dir or selfreport_dir()
    if synthetic:
        df, task_df, meta = build_table(synthetic_parsed(), log=log)
        meta["synthetic"] = True
    else:
        files = find_subject_files(roots)
        if not files:
            raise SelfReportError("nessun After_remarks.mat/.pkl trovato negli input: aggiungi i file "
                                  "comportamentali di FACED al Dataset dei dati")
        parsed, struct = {}, {}
        for s, p in files.items():
            parsed[s] = parse_after_remarks(read_any(p))
            if len(struct) < 3:
                struct[f"sub{s:03d}"] = {"file": str(p), "fields": parsed[s]["fields"]}
                log(f"[voti] sub{s:03d} {p.name}: {parsed[s]['fields']}")
        df, task_df, meta = build_table(parsed, log=log)
        meta["structure_first3"] = struct
        meta["files"] = {"n": len(files), "formats": sorted({p.suffix for p in files.values()})}
        meta["stimuli_info"] = check_stimuli_info(roots)
        meta["structure_xlsx"] = describe_structure_file(roots)
    meta = write_outputs(df, task_df, meta, out_dir, log=log)
    verify_hash(meta)
    return meta


# ------------------------------------------------------------------ lettura della tabella e sintetico
_CACHE: dict = {}


def load_votes(path: Path | None = None, synthetic: bool | None = None) -> pd.DataFrame:
    """Tabella dei voti dallo stato (scritta dal passo 0). In modalità sintetica la genera al volo."""
    import os
    synthetic = (os.environ.get("PIANOB_SYNTHETIC") == "1") if synthetic is None else synthetic
    key = (str(path), synthetic)
    if key in _CACHE:
        return _CACHE[key]
    if path is None:
        d = selfreport_dir()
        path = d / "selfreport.parquet" if (d / "selfreport.parquet").exists() else d / "selfreport.csv"
    if Path(path).exists():
        df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    elif synthetic:
        df, _, _ = build_table(synthetic_parsed(), log=lambda *a: None)
    else:
        raise FileNotFoundError(f"tabella dei voti {path} assente: esegui prima il passo 0 (run_step0)")
    _CACHE[key] = df
    return df


def synthetic_parsed(n_subjects: int = FM.N_SUBJECTS, seed: int = 0, presentation_rows: bool = True,
                     individual: float = 0.5) -> dict[int, dict]:
    """Voti sintetici plausibili: item della propria categoria alti, valenza dal segno della clip, offset di
    soggetto, componente individuale (individual) e rumore; righe in ordine di presentazione con vid permutato."""
    rng = np.random.default_rng(seed)
    clip_eff = rng.normal(0, 0.6, (28, 12))
    out = {}
    for s in range(n_subjects):
        srng = np.random.default_rng(seed * 10_000 + s)
        offs = srng.normal(0, 0.7, 12)
        sc = np.full((28, 12), 2.0) + clip_eff + offs
        for c in range(28):
            cat = FM.CLIP_LABELS[c]
            for it, k in ITEM_CATEGORY.items():
                if k == cat:
                    sc[c, ITEMS.index(it)] += 3.0
            sc[c, ITEMS.index("valence")] += 2.0 * FM.CLIP_VALENCE[c] + 1.5
            sc[c, ITEMS.index("arousal")] += 1.5 * (FM.CLIP_VALENCE[c] != 0)
        sc += individual * srng.normal(0, 1, (28, 12)) + srng.normal(0, 0.5, (28, 12))
        sc = np.clip(np.round(sc), 0, 7)
        order = srng.permutation(28)                    # ordine di presentazione
        rows = order if presentation_rows else np.arange(28)
        trial = np.argsort(order) + 1 if not presentation_rows else np.arange(1, 29)
        out[s] = {"score": sc[rows], "trial": trial.astype(int), "vid": rows + 1, "extra": {},
                  "fields": {"score": "(28, 12)", "trial": "(28,)", "vid": "(28,)"}}
    return out
