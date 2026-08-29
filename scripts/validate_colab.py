"""Validate the Python code cells of the Colab notebook compile correctly."""
import ast
import json
from pathlib import Path

nb = json.loads(Path(r"notebooks/question_answering_colab.ipynb").read_text(encoding="utf-8"))
ok = True
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # Strip Colab shell/line-magic lines ('!', '%') which are not Python.
    lines = [l for l in src.split("\n")
             if not l.strip().startswith("!") and not l.strip().startswith("%")]
    code = "\n".join(lines)
    try:
        ast.parse(code)
        print(f"cell {i}: OK")
    except SyntaxError as e:
        ok = False
        print(f"cell {i}: SYNTAX ERROR -> {e}")

print("ALL_CODE_CELLS_OK" if ok else "HAD_ERRORS")
