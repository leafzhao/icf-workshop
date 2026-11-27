# ICF data-analysis workshop

Two independent sessions with real laser-plasma data:

1. **Laser pulse shape (OMEGA shot 93305).** Students analyse P510 streak-camera data, from the raw CCD image to the average pulse and its statistics.
2. **Time-resolved scattered-light spectrum (tSPM, 2021-08-14 shot 110).** Students take a streak-spectrometer image to a publication-style figure: axes, background, GVD correction, colour scale and colour map, laser overlay, and lineouts of the SRS and ω/2 bands.

## Files

- `pulse_workshop.ipynb` and `tspm_workshop.ipynb` are the notebooks students run, one step per cell.
- `pulse.py` and `tspm.py` hold the helper functions each notebook calls.
- `data/` holds the P510 shot data. The pulse notebook reads only the `.h5` files; the `.hdf` files are HDF4 copies with the same content.
- `data/time-resolved spec/` holds the tSPM data: `20210814110.dac` (the spectrum, a text table exported by the streak camera) and `20210814110_2X_1.csv` to `_8.csv` (oscilloscope traces of beams 1 to 8).

The data files are not in this repository. Get them from the workshop organisers and put them in `data/` with the layout above before you run the notebooks.

## Setup (Windows, macOS, Linux)

Install Python 3.10 or newer (for example from python.org or Anaconda), then in a terminal:

```bash
pip install -U -r requirements.txt
```

`requirements.txt` lists numpy, scipy, matplotlib, h5py and jupyterlab. Tested with Python 3.12, numpy 2.5.0, scipy 1.18.1, matplotlib 3.11.0, h5py 3.16.0 and jupyterlab 4.3.4.

Keep `-U`: it upgrades numpy and scipy together. An old scipy next to a new numpy prints `A NumPy version >=... and <... is required for this version of SciPy` and may give wrong results. If you see it, run the command above again and restart the kernel.

`scripts/compare.py` also needs `pyhdf` to read the HDF4 files; the notebooks do not.

```bash
jupyter lab pulse_workshop.ipynb
```

```bash
jupyter lab tspm_workshop.ipynb
```

Run the cells from top to bottom with `Shift + Enter`.

No install: upload the notebook, its helper file (`pulse.py` or `tspm.py`) and the `data` folder to Google Colab and run the notebook there.

## Known limits

- Pulse shape: raw images exist for clusters 1 and 2 only (beams 10 to 29). The 60-beam result comes from the facility's processed file.
- tSPM: the spectrum's time zero is set by hand (`T0 = 1.05` ns, from the original analysis), not measured. The laser traces are normalised per beam, because the beam energies are not included.
