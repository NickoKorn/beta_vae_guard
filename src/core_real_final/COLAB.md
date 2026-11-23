# Guarded Varix in Google Colab

Activate a GPU via **Runtime → Change runtime type → T4 GPU**. Run each block in a Colab cell.

```python
!git clone https://github.com/NickoKorn/beta_vae_guard.git
%cd /content/beta_vae_guard
%pip install -r src/core_real_final/requirements-colab.txt
```

Colab environments can change. The experiment was developed with Python 3.11 and the pinned AUTOENCODIX commit above. Select a GPU runtime before executing the next cell.

Download the TCGA archive from the authors' Zenodo record and extract the three parquet files. The archive requires several gigabytes of free storage. Downloads are subject to the source's terms and availability.

```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import json

config = json.loads(Path('src/core_real_final/guarded_varix_colab.json').read_text())
tcga = config['data']['tcga']
folder = Path(tcga['root'])
folder.mkdir(parents=True, exist_ok=True)
archive = Path('/content/autoencodix_tcga.zip')
urlretrieve('https://zenodo.org/api/records/15518831/files/AutoencodixZenodoReproducibility_v2.zip/content', archive)
with ZipFile(archive) as data:
    for member in tcga['source']['members'].values():
        data.extract(member['name'], folder)
        assert (folder / member['name']).stat().st_size == member['size']
archive.unlink()
```

Mount Drive before a long run, then set an output folder. Save outputs to Drive because Colab local storage is temporary. The run directory must be new.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
output = Path('/content/drive/MyDrive/beta_guard_results')
output.mkdir(parents=True, exist_ok=True)
```

Run a single TCGA model first. This example uses the saved 100-epoch PI configuration with `ki=0.06` per step, `kp=0.002`, and Lucas epsilon 0.2. The configured 1000 epochs refer to the annealing budget and the 100 control epochs start after handover.

```python
!python -m src.core_real_final.main "{output}" --config src/core_real_final/guarded_varix_colab.json --group weak_signal_lowvar --variant base --split-seed 1 --train-seed 1
```

List the 432 planned TCGA runs with `--list-grid`. To run them sequentially in a live session, use `--grid` with the same config and output folder. A Colab runtime may stop before a long grid completes. The standard runner stops when a run directory already exists, so use a fresh output directory for a grid.

For circle or sphere data, run the same entry point with `--dataset gudhi --topology circle` or `--topology sphere`. The data are generated locally and cached under `data/gudhi`.
