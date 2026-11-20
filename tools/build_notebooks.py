"""notebooks_src/*.py (formato percent `# %%`) -> notebooks/*.ipynb con nbformat (jupytext non serve).

    python tools/build_notebooks.py            # genera notebooks/*.ipynb
    python tools/build_notebooks.py --flat DIR # genera anche script .py eseguibili (per prove locali)

`# %% INCLUDE file.py` inserisce le celle di notebooks_src/file.py.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_src"


def parse_cells(path: Path) -> list[tuple[str, str]]:
    cells, kind, buf = [], None, []

    def flush():
        if kind is not None:
            text = "\n".join(buf).strip("\n")
            if text:
                cells.append((kind, text))

    for line in path.read_text().splitlines():
        m = re.match(r"^# %%(.*)$", line)
        if m:
            flush()
            header = m.group(1).strip()
            buf = []
            inc = re.match(r"INCLUDE\s+(\S+)", header)
            if inc:
                cells.extend(parse_cells(path.parent / inc.group(1)))
                kind = None
                continue
            kind = "markdown" if "[markdown]" in header else "code"
            continue
        if kind == "markdown":
            line = re.sub(r"^# ?", "", line)
        if kind is not None:
            buf.append(line)
    flush()
    return cells


def build(path: Path, out_dir: Path, flat_dir: Path | None = None) -> Path:
    cells = parse_cells(path)
    nb = nbformat.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [nbformat.v4.new_markdown_cell(t) if k == "markdown" else nbformat.v4.new_code_cell(t)
                for k, t in cells]
    nbformat.validate(nb)
    out = out_dir / f"{path.stem}.ipynb"
    nbformat.write(nb, out)
    if flat_dir is not None:
        flat_dir.mkdir(parents=True, exist_ok=True)
        (flat_dir / f"{path.stem}_flat.py").write_text("\n\n".join(t for k, t in cells if k == "code") + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "notebooks"))
    ap.add_argument("--flat", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(SRC.glob("nb*.py")):
        o = build(p, out, Path(a.flat) if a.flat else None)
        print(f"{p.name} -> {o.relative_to(ROOT) if o.is_relative_to(ROOT) else o}")


if __name__ == "__main__":
    main()
