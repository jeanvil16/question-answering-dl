"""Patch the Colab notebook cell 5:
- add `import sys`,
- explicitly insert PROJECT_DIR into sys.path so `import model / preprocessing`
  works in Colab even though os.chdir alone may not update sys.path.
"""
import json
from pathlib import Path

nb = json.loads(Path(r"notebooks/question_answering_colab.ipynb").read_text(encoding="utf-8"))

src = "".join(nb["cells"][5]["source"])

# 1. ensure sys is imported in this cell
if "\nimport sys\n" not in src:
    src = src.replace("import os\nimport shutil\n", "import os\nimport sys\nimport shutil\n",
                      count=1)

# 2. insert PROJECT_DIR into sys.path right after os.chdir
if "sys.path.insert(0, str(PROJECT_DIR))" not in src:
    anchor = 'os.chdir(PROJECT_DIR)\n'
    src = src.replace(
        anchor,
        anchor + '# Ensure the project root is importable (os.chdir alone may not update sys.path).\nsys.path.insert(0, str(PROJECT_DIR))\n',
        count=1,
    )

nb["cells"][5]["source"] = src

Path(r"notebooks/question_answering_colab.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("cell 5 patched")
print("---- new cell 5 ----")
print(src)
