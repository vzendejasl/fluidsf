# FluidSF Setup

This repository can be used with either `conda` or a local virtual environment, but the recommended workflow for this project is Conda.

## Conda Setup

From the repository root:

```bash
cd /Users/victorzendejaslopez/Documents/fluidsf
conda env create -f environment.yaml
conda activate fluidsf-dev
```

This environment file installs:

- Python 3.11
- the local `fluidsf` package in editable mode
- test dependencies
- example and notebook dependencies
- `jupyterlab`

If the environment already exists, update it instead:

```bash
cd /Users/victorzendejaslopez/Documents/fluidsf
conda env update -f environment.yaml --prune
conda activate fluidsf-dev
```

If `conda activate` does not work in your shell, initialize Conda first:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fluidsf-dev
```

## Run Example Scripts

Run a single example:

```bash
python examples/python_scripts/ex_1d.py
```

Other examples:

```bash
python examples/python_scripts/ex_2d.py
python examples/python_scripts/ex_3d.py
python examples/python_scripts/qs.py
python examples/python_scripts/ex_cascade.py
python examples/python_scripts/ex_2d_bootstrap.py
python examples/python_scripts/maps_2d_ex.py
```

## Run Jupyter Notebooks

Start JupyterLab:

```bash
jupyter lab
```

Then open any notebook in [examples/jupyter_notebooks](/Users/victorzendejaslopez/Documents/fluidsf/examples/jupyter_notebooks), for example [examples/jupyter_notebooks/ex_1d.ipynb](/Users/victorzendejaslopez/Documents/fluidsf/examples/jupyter_notebooks/ex_1d.ipynb).

To run notebook cells:

- use `Shift+Enter` for the current cell
- or use the JupyterLab menu to run all cells

## Run Tests

Run the full test suite:

```bash
pytest tests/
```

Run one test file:

```bash
pytest tests/test_bin_data.py
```

Run the MPI-enabled integration tests:

```bash
python -m pip install -e '.[mpi,test]'
export FLUIDSF_RUN_MPI_TESTS=1
export HYDRA_LAUNCHER=fork
python -m pytest tests/test_mpi_integration_generate_sf_3d.py
```

Run the public 3D MPI backend manually:

```bash
mpirun -launcher fork -n 4 python -c "import numpy as np, fluidsf; x=np.arange(8,dtype=float); y=np.arange(8,dtype=float); z=np.arange(8,dtype=float); u,v,w=np.meshgrid(x,y,z,indexing='ij'); sf=fluidsf.generate_structure_functions_3d(u,v,w,x,y,z,sf_type=['LL','TT','LLL','LTT'],boundary='periodic-all',backend='mpi',px=2); print(sf['SF_LL_y'] if sf['x-diffs'] is not None else 'worker rank')"
```

Current MPI notes:

- 1D `backend="mpi"` distributes separation work, but each rank still stores the full input arrays
- 2D `backend="mpi"` supports distributed uniform-grid x-slabs shaped `(len(y), local_x)` and falls back to the older full-array path for unsupported cases
- 3D `backend="mpi"` supports distributed slabs and is the path intended for larger 3D runs
- the 3D backend supports `boundary=None`, `periodic-all`, and mixed periodic boundary combinations

Benchmark the public 3D backend with:

```bash
python benchmarks/benchmark_3d_scaling.py --backend serial
mpirun -launcher fork -n 4 python benchmarks/benchmark_3d_scaling.py --backend mpi --layout distributed
```

Benchmark the public 2D backend with:

```bash
python benchmarks/benchmark_2d_scaling.py --backend serial
mpirun -launcher fork -n 4 python benchmarks/benchmark_2d_scaling.py --backend mpi --layout distributed
```

## Quick Smoke Test

After setup, this sequence should work:

```bash
python -c "import fluidsf; print(fluidsf.__version__)"
python examples/python_scripts/ex_1d.py
pytest tests/test_bin_data.py
```
