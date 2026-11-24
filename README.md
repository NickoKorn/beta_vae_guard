# Beta Guard VAE

Closed-loop ControlVAE-PI training for `Varix` (AUTOENCODIX) with Lucas $(\epsilon, \delta)$-collapse detection and ring-buffer rollback.

## Struktur

- [`configs/`](configs/) – Experiment-Konfigurationen (`guarded_varix.json`, `guarded_varix_eps0_1.json`)
- [`src/core_real_final/`](src/core_real_final/) – Trainer, PI-Scheduler, Datenloader & CLI
- [`tests/`](tests/) – Unit- & Regressionstests
- [`windows/`](windows/) – PowerShell-Startskripte & Windows-Requirements (RTX 5090 / CUDA 12.8)
- [`notebooks/`](notebooks/) – Auswertungs-Notebooks (u. a. `kneedle_rd_analysis.ipynb`)

## Starten (Linux)

Umgebung einrichten (`uv`):
```bash
uv sync
```

Einzellauf (TCGA oder GUDHI):
```bash
.venv/bin/python -m src.core_real_final.main /pfad/zum/output --dataset tcga --group ontix_standard --variant base
.venv/bin/python -m src.core_real_final.main /pfad/zum/output --dataset gudhi --topology circle --ki 0.6
```

Komplettes Raster anzeigen oder starten:
```bash
.venv/bin/python -m src.core_real_final.main --list-grid --dataset tcga
.venv/bin/python -m src.core_real_final.main /pfad/zum/output --dataset tcga --grid
```

TCGA-Baselines (`freeze`, `early_stop`, `controlvae`) + GUDHI-Suite:
```bash
.venv/bin/python -m src.core_real_final.run_5090_experiments /pfad/zum/output --suite all --workers 2
```

## Starten (Windows / PowerShell)

Aus dem Repo-Hauptordner (richtet `.venv-windows` mit PyTorch CUDA 12.8 automatisch ein):
```powershell
# Baselines + GUDHI (eps = 0.1):
powershell -ExecutionPolicy Bypass -File .\windows\run_5090_experiments.ps1

# Paralleles TCGA-Raster (4 Worker, eps = 0.01):
powershell -ExecutionPolicy Bypass -File .\windows\run_windows_parallel4.ps1
```

## Tests

```bash
PYTHONPATH=. .venv/bin/pytest -q tests/test_activation_window.py
```
