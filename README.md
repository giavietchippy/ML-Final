# ML-Final

Final machine learning project.



## Q2a quick start

Copy the entire `Q2a/` folder to the new machine, including `results/`. Then, from inside `Q2a/`:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace Q2a_CNN.ipynb
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace Q2a_CNN.ipynb
```

Or open `Q2a_CNN.ipynb` in Jupyter/VS Code and select the kernel from the `.venv` you created.

All three training switches in Section 1 are `False` by default, so the notebook performs no training and reads the saved artifacts in `results/`. If you enable a switch, all generated files go only to `results_nb/`; `results/` is never written.

---

## CIFAR-10 cache

The notebook uses `keras.datasets.cifar10.load_data()`. To avoid a slow download on a fresh machine, copy the CIFAR-10 cache from a machine that already has it:

| OS | Cache folder |
|---|---|
| Windows | `%USERPROFILE%\.keras\datasets\` |
| macOS / Linux | `$HOME/.keras/datasets/` |

The cache should contain `cifar-10-batches-py-target/` and `cifar-10-batches-py-target_archive`. The cache is per user account, not per Python environment.

Colab is not recommended because it starts with an empty cache.

---

## Path and dependency issues

If the notebook cannot locate the project folder, set `Q2A_PROJECT_DIR` before the setup cell:

```python
import os
os.environ["Q2A_PROJECT_DIR"] = r"D:\path\to\Q2a"   # Windows
# os.environ["Q2A_PROJECT_DIR"] = "/path/to/Q2a"    # macOS/Linux
```

If the kernel reports a missing package, install the requirements into that kernel, or register the project environment:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name q2a-venv
```

Then select the `q2a-venv` kernel.

See [`Q2a/README.md`](Q2a/README.md) for the complete guide, troubleshooting, dependency lock notes, and explanation of the original scripts.
