# EXP, NNQ, and NNQSM Hydrological Models

This repository contains the EXP conceptual model and two hybrid hydrological models with embedded neural networks (ENNs):

| Model | Entry script | Description |
| --- | --- | --- |
| EXP | `EXP/EXP.py` | Conceptual hydrological model with physical process formulations. |
| NNQ | `NN_Q/NNQ.py` | Hybrid model replacing runoff generation with an ENN. |
| NNQSM | `NN_QSM/NNQSM.py` | Hybrid model replacing runoff generation, precipitation partitioning, and snowmelt with ENNs. |


## Acknowledgements

We gratefully acknowledge Jiang, Zheng, and Solomatine for developing and sharing the original hydrology-aware deep learning model. The programs in this repository were adapted from their [physics-aware-dl implementation](https://github.com/oreopie/physics-aware-dl), using the v1.2 source package as the starting point.

Please acknowledge the original work when using this code:

Jiang, S., Zheng, Y., and Solomatine, D. (2020). *Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning*. Geophysical Research Letters, 47, e2020GL088229. [https://doi.org/10.1029/2020GL088229](https://doi.org/10.1029/2020GL088229).

## Python environment

Use a dedicated environment with the core versions specified in the original project's setup instructions:

| Component | Version |
| --- | --- |
| Python | 3.6 |
| TensorFlow | 1.14.0 |
| Standalone Keras | 2.3.1 |
| NumPy | 1.16.4 |

An example setup using Conda is:

```bash
conda create -n hybrid-hydro python=3.6
conda activate hybrid-hydro
python -m pip install numpy==1.16.4 tensorflow==1.14.0 keras==2.3.1 h5py==2.10.0 pandas scipy openpyxl==3.0.10
```

`h5py` supports the HDF5 weight files, and `openpyxl` supports Excel result export. This setup follows the original project's legacy TensorFlow/Keras environment; it is not a newly tested environment specification for this repository. The scripts use standalone `keras` and TensorFlow 1.x APIs.

## Download and prepare CAMELS data

Download the **CAMELS-US meteorological forcing and observed streamflow time series** from the [official CAMELS dataset download page](https://ral.ucar.edu/dataset/camels-catchment-attributes-and-meteorology-large-sample-studies-dataset-downloads). The original implementation uses the `basin_timeseries_v1p2_metForcing_obsFlow.zip` archive and **Daymet** forcing.

Extract and arrange the data beneath your chosen `working_path`:

```text
working_path/
├── basin_list_cold.txt
├── basin_list.txt
└── camels/
    ├── basin_mean_forcing/
    │   └── daymet/
    │       ├── 01/
    │       │   └── 01013500_lump_cida_forcing_leap.txt
    │       ├── ...
    │       └── 18/
    └── usgs_streamflow/
        ├── 01/
        │   └── 01013500_streamflow_qc.txt
        ├── ...
        └── 18/
```

The repository's `basin_list_cold.txt` selects the 31 study basins. The data loader separately reads `basin_list.txt`, which maps basin IDs to HUC directories. Both files must be tab-separated and contain the columns `HUC` and `BASIN_ID`; preserve leading zeros in basin IDs.

To prepare both lists for the supplied study basins, run the following Python snippet from the repository root after changing the example data directory:

```python
from pathlib import Path
import pandas as pd

working_path = Path('/path/to/CAMELS/DATA')
working_path.mkdir(parents=True, exist_ok=True)

basins = pd.read_csv('basin_list_cold.txt', sep='\t', dtype=str)
basins.to_csv(working_path / 'basin_list_cold.txt', sep='\t', index=False)
basins['HUC'] = basins['HUC'].str.zfill(2)
basins.to_csv(working_path / 'basin_list.txt', sep='\t', index=False)
```

This produces HUC values such as `01` matching the data directories. Ensure that the selected basins have both forcing and streamflow files. Please cite the dataset following the guidance on the official download page.

## Configure paths and run

Before running, edit **both** path variables inside the `models()` function in each entry script you intend to use:

```python
working_path = '/path/to/CAMELS/DATA/'
results_path = '/path/to/model_outputs/'
```

`working_path` is the parent directory containing `camels/` and the two basin lists. `results_path` is the parent directory where model weights and simulation results will be saved. Replace the server-specific defaults with directories on your own machine. On Windows, paths such as `D:/CAMELS/DATA/` can be used.

Run each model from its own directory so that its local support modules are imported:

```bash
# From the repository root:
cd EXP
python EXP.py
cd ..

cd NN_Q
python NNQ.py
cd ..

cd NN_QSM
python NNQSM.py
cd ..
```

The supplied scripts run 50 ensemble members for the 31 study basins and use eight worker processes. Adjust `Pool(8)` to match available computing resources. Training covers 1980-10-01 to 2000-09-30; the reported evaluation period is 2000-10-01 to 2010-09-30, following model spin-up.

## Outputs

The scripts create model-specific subdirectories under `results_path`:

```text
results_path/
├── EXP/
├── NNQ/
└── NNQSM/
```

Each model directory contains `MODEL/NUM1/` through `MODEL/NUM50/` for `.h5` weights and `RESULTS/NUM1/` through `RESULTS/NUM50/` for per-basin `.xlsx` simulations. Result columns contain observed and simulated runoff, precipitation partitioning, snow and soil storage, snowmelt, and evapotranspiration; EXP additionally exports subsurface and surface runoff.
