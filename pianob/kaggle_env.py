"""Ambiente di esecuzione: Kaggle o locale, percorsi, GPU, guardia del tempo, ripresa."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
STATE_DIRNAME = "pianob_state"
# Sottocartelle dello stato che si portano da una sessione all'altra.
PERSISTENT_SUBDIRS = ["results", "done", "ckpt", "optuna", "hpo_best", "models", "bench", "artifacts", "logs"]


def on_kaggle() -> bool:
    return KAGGLE_WORKING.exists() and KAGGLE_INPUT.exists()


def working_dir() -> Path:
    if os.environ.get("PIANOB_WORKING"):
        p = Path(os.environ["PIANOB_WORKING"])
    elif on_kaggle():
        p = KAGGLE_WORKING
    else:
        p = Path(__file__).resolve().parents[1] / "local_work"
    p.mkdir(parents=True, exist_ok=True)
    return p


def input_roots() -> list[Path]:
    roots = []
    if os.environ.get("PIANOB_INPUT"):
        roots += [Path(p) for p in os.environ["PIANOB_INPUT"].split(os.pathsep) if p]
    if on_kaggle():
        roots.append(KAGGLE_INPUT)
    return [r for r in roots if r.exists()]


def state_dir() -> Path:
    p = working_dir() / STATE_DIRNAME
    for sub in PERSISTENT_SUBDIRS + ["locks"]:
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    """Cache ricalcolabili (dati filtrati, feature): su Kaggle fuori da /kaggle/working, così non
    finiscono negli output della versione."""
    p = Path("/tmp/pianob_cache") if on_kaggle() else state_dir() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_file(name: str, roots: list[Path] | None = None, max_depth: int = 6) -> Path | None:
    """Ricerca ricorsiva (la struttura annidata dei Dataset Kaggle può cambiare)."""
    roots = roots if roots is not None else input_roots() + [working_dir()]
    for root in roots:
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            if name in filenames:
                return Path(dirpath) / name
    return None


def find_all(name: str, roots: list[Path] | None = None, max_depth: int = 7) -> list[Path]:
    roots = roots if roots is not None else input_roots()
    out = []
    for root in roots:
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            if name in filenames:
                out.append(Path(dirpath) / name)
    return out


def find_dir(name: str, roots: list[Path] | None = None, max_depth: int = 7) -> list[Path]:
    roots = roots if roots is not None else input_roots()
    out = []
    for root in roots:
        base_depth = len(root.parts)
        for dirpath, dirnames, _ in os.walk(root):
            if len(Path(dirpath).parts) - base_depth >= max_depth:
                dirnames[:] = []
            if name in dirnames:
                out.append(Path(dirpath) / name)
    return out


def restore_state_from_inputs(verbose: bool = True) -> dict:
    """Copia nello stato corrente risultati, done, checkpoint, .db Optuna ecc. trovati negli
    output di sessioni precedenti collegati come input. Non sovrascrive file più recenti."""
    dst = state_dir()
    copied = {}
    for src_state in find_dir(STATE_DIRNAME):
        if dst.resolve() == src_state.resolve():
            continue
        for sub in PERSISTENT_SUBDIRS:
            s = src_state / sub
            if not s.exists():
                continue
            for f in s.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(src_state)
                t = dst / rel
                if t.exists() and t.stat().st_size >= f.stat().st_size and sub not in ("results", "done"):
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                if sub in ("results", "done") and t.exists():
                    # file jsonl: unione delle righe (stesso nome = stesso worker di una sessione precedente)
                    lines = set(t.read_text().splitlines()) | set(f.read_text().splitlines())
                    t.write_text("\n".join(sorted(l for l in lines if l.strip())) + "\n")
                else:
                    shutil.copy2(f, t)
                copied[sub] = copied.get(sub, 0) + 1
    if verbose:
        print(f"[ripresa] file recuperati da sessioni precedenti: {copied or 'nessuno'}")
    return copied


def clear_locks() -> None:
    for f in (state_dir() / "locks").glob("*.lock"):
        f.unlink(missing_ok=True)


def gpu_info() -> list[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"index": i, "name": p.name, "mem_gb": round(p.total_memory / 2**30, 1),
                        "cc": f"{p.major}.{p.minor}"})
        return out
    except Exception:
        return []


def describe_env() -> dict:
    import platform
    info = {"kaggle": on_kaggle(), "python": platform.python_version(), "cpus": os.cpu_count(),
            "gpus": gpu_info(), "working": str(working_dir())}
    for mod in ("torch", "numpy", "scipy", "sklearn", "pandas", "optuna"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return info


def code_version() -> str:
    root = Path(__file__).resolve().parents[1]
    v = root / "VERSION"
    if v.exists():
        return v.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        from . import __version__
        return __version__


class TimeGuard:
    """Guardia del tempo per la sessione Kaggle (12 h): si ferma in modo pulito prima.

    Il tempo zero è letto da PIANOB_T0 (impostato dal notebook all'avvio), altrimenti è l'import.
    """

    def __init__(self, budget_hours: float | None = None, margin_min: float = 10.0):
        self.t0 = float(os.environ.get("PIANOB_T0", time.time()))
        if budget_hours is None:
            budget_hours = float(os.environ.get("PIANOB_BUDGET_H", 11.0 if on_kaggle() else 1e6))
        self.budget_s = budget_hours * 3600
        self.margin_s = margin_min * 60

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60

    def remaining_min(self) -> float:
        return (self.budget_s - (time.time() - self.t0)) / 60

    def can_start(self, est_minutes: float) -> bool:
        return self.remaining_min() - self.margin_s / 60 > est_minutes * 1.15

    def must_stop(self) -> bool:
        return self.remaining_min() < self.margin_s / 60


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)
