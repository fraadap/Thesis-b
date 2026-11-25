"""Pesi di REVE-Base: download a runtime da Hugging Face (repo con condizioni d'uso) e caricamento strict.

I pesi NON si mettono nello zip dei dati né negli output dei notebook (condizioni d'uso: niente
redistribuzione). Su Kaggle il token HF sta in un Kaggle Secret (`HF_TOKEN`); il notebook G lo legge, scarica
`model.safetensors` e `config.json` nella cache in /tmp (fuori da /kaggle/working, così i pesi non finiscono
nell'output della sessione che poi si ricarica come Dataset) e annota la revisione risolta.

Il formato safetensors si legge senza dipendenze (header JSON + buffer), se il pacchetto `safetensors` manca.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import urllib.error
import urllib.request
from pathlib import Path

import torch

REPO = "brain-bzh/reve-base"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def cache_dir() -> Path:
    from ... import kaggle_env
    d = Path(os.environ.get("PIANOB_REVE_CACHE", str(kaggle_env.cache_dir() / "reve")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, token: str | None, dest: Path | None = None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=600) as r:
        if dest is None:
            return r.read(), dict(r.headers)
        tmp = Path(str(dest) + ".part")
        with open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
        return None, dict(r.headers)


def resolve_revision(token: str | None, revision: str = "main") -> str:
    """Commit della revisione richiesta (da annotare nei risultati)."""
    body, _ = _get(f"https://huggingface.co/api/models/{REPO}/revision/{revision}", token)
    return json.loads(body)["sha"]


def download(token: str | None, revision: str | None = None, log=print) -> dict:
    """Scarica config e pesi nella cache. `revision`: commit da fissare (default: env PIANOB_REVE_REVISION,
    altrimenti 'main' risolta in un commit che viene annotato)."""
    revision = revision or os.environ.get("PIANOB_REVE_REVISION", "main")
    sha = resolve_revision(token, revision)
    d = cache_dir() / sha
    d.mkdir(parents=True, exist_ok=True)
    cfg_body, _ = _get(f"https://huggingface.co/{REPO}/resolve/{sha}/config.json", token)
    (d / "config.json").write_bytes(cfg_body)
    path = None
    for fname in WEIGHT_FILES:
        dest = d / fname
        if dest.exists():
            path = dest
            break
        try:
            _get(f"https://huggingface.co/{REPO}/resolve/{sha}/{fname}", token, dest)
            path = dest
            break
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    if path is None:
        raise FileNotFoundError(f"nessun file di pesi {WEIGHT_FILES} in {REPO}@{sha}")
    info = {"repo": REPO, "revision_requested": revision, "revision": sha, "file": str(path),
            "sha256": sha256(path), "config": json.loads(cfg_body)}
    (d / "download_info.json").write_text(json.dumps(info, indent=1))
    os.environ["PIANOB_REVE_WEIGHTS"] = str(path)
    log(f"[REVE] pesi {REPO}@{sha} -> {path} (sha256 {info['sha256'][:12]}...)")
    return info


def find_weights() -> Path | None:
    p = os.environ.get("PIANOB_REVE_WEIGHTS")
    if p and Path(p).exists():
        return Path(p)
    for f in WEIGHT_FILES:
        hits = sorted(cache_dir().glob(f"*/{f}"))
        if hits:
            return hits[-1]
    return None


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64,
       "I64": torch.int64, "I32": torch.int32, "BOOL": torch.bool}


def read_safetensors(path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
        return load_file(str(path))
    except ImportError:
        pass
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        buf = bytearray(raw[base + s:base + e])
        t = torch.frombuffer(buf, dtype=_DT[meta["dtype"]]) if buf else torch.empty(0, dtype=_DT[meta["dtype"]])
        out[name] = t.reshape(meta["shape"]).clone()
    return out


def load_state(path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        return read_safetensors(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("state_dict", sd)


def check_config(cfg: dict, arch: dict) -> None:
    """La config del checkpoint deve coincidere con l'architettura vendorizzata."""
    for k in ("embed_dim", "depth", "heads", "head_dim", "mlp_dim_ratio", "use_geglu", "freqs", "noise_ratio",
              "patch_size", "patch_overlap"):
        if k in cfg:
            assert cfg[k] == arch[k], f"config REVE: {k}={cfg[k]} diverso da {arch[k]}"


def load_into(model, path) -> dict:
    """Carica l'encoder (strict) e la query `cls` dal checkpoint HF (`REVE.from_pretrained` + `dt.py`).
    Pesi mancanti o chiavi inattese = errore (§R8.9: nel codice ufficiale era solo un warning)."""
    sd = load_state(path)
    for pref in ("module.", "model.", "reve."):
        if sd and all(k.startswith(pref) for k in sd):
            sd = {k[len(pref):]: v for k, v in sd.items()}
    cls = sd.pop("cls_query_token", None)
    if cls is None:
        raise RuntimeError("cls_query_token assente nel checkpoint REVE")
    own = model.encoder.state_dict()
    missing, unexpected = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
    if missing or unexpected:
        raise RuntimeError(f"pesi REVE non corrispondenti: mancanti {missing[:8]} ({len(missing)}), "
                           f"inattesi {unexpected[:8]} ({len(unexpected)})")
    model.encoder.load_state_dict(sd, strict=True)
    with torch.no_grad():
        model.cls_query_token.copy_(cls.reshape(model.cls_query_token.shape))
    info = {"path": str(path), "sha256": sha256(path), "n_tensors": len(sd) + 1, "strict": True,
            "cls_query_token": "caricata"}
    meta = Path(path).parent / "download_info.json"
    if meta.exists():
        dl = json.loads(meta.read_text())
        info["revision"] = dl.get("revision")
        check_config(dl.get("config", {}), model_arch(model))
    return info


def model_arch(model) -> dict:
    from .model import REVE_BASE
    enc = model.encoder
    return {**REVE_BASE, "embed_dim": enc.embed_dim, "freqs": enc.freqs, "patch_size": enc.patch_size,
            "patch_overlap": enc.overlap_size, "noise_ratio": enc.noise_ratio,
            "depth": len(enc.transformer.layers)}
