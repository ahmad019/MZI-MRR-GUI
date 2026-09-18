# MZI Ring Resonator Analysis Tool

A desktop GUI for simulating and visualizing the optical response of a Mach-Zehnder Interferometer (MZI) coupled to one or two ring resonators. Built with Python, Tkinter, and SymPy/NumPy for the underlying physics, it lets you sweep parameters interactively and see the output intensity, phase, group index, and group velocity update live.

## Features

- **Three coupling configurations**
  - **Config A** — both resonators side-coupled to the waveguide (transmission mode)
  - **Config B** — resonator 1 side-coupled, resonator 2 inserted in the optical path (mixed transmission/reflection)
  - **Config C** — both resonators inserted in the optical path (reflection mode)
- **Interactive parameter controls** — sliders with live numeric entry for wavelength, ring radius, coupling coefficients (r1–r4), quality factors (Q1, Q2), material attenuation, and phase shifters (η1, η2)
- **Real-time plots** (6-panel view): total output intensity, output phase, individual resonator phases, individual transmissions, group index, and group velocity
- **Smart zoom** — a second window that automatically zooms into the regions of highest variation (e.g. near resonances)
- **Data export** — save the current sweep results and parameters to CSV or Excel (`.xlsx`)
- **Parameter validation** — flags physically inconsistent or out-of-range parameter combinations
- **Built-in help** — a User Guide window (Help menu) explaining configurations, parameters, and coupling states

## Requirements

- Python 3.8+
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [sympy](https://www.sympy.org/)
- [pandas](https://pandas.pydata.org/)
- [openpyxl](https://openpyxl.readthedocs.io/) (only needed for `.xlsx` export)

Tkinter ships with most standard Python installations. On some Linux distributions you may need to install it separately (e.g. `sudo apt install python3-tk`).

## Installation

```bash
git clone https://github.com/ahmad019/MZI-MRR-GUI.git
cd MZI-MRR-GUI
pip install -r requirements.txt
```

## Usage

Run the program with:

```bash
python mzi_gui.py
```

This opens the main window with a control panel on the left and a 6-panel plot view on the right.

1. Choose a **Configuration** (A, B, or C).
2. Adjust the **Physical Parameters** (wavelength, effective index, ring radii) and per-resonator settings (coupling r, Q factor, attenuation, resonant wavelength).
3. Set the **Phase Shifters** (η1, η2) and the **Sweep Range** for φ1.
4. Click **Update Plots** to recalculate, **Show Zoomed** for a focused view near resonances, **Export Data** to save results, or **Reset Parameters** to restore defaults.

## Configurations Explained

| Config | Description | Notes |
|---|---|---|
| A | Both resonators side-coupled | Uses transmission phase/amplitude for both arms |
| B | Resonator 1 coupled, resonator 2 inserted | Requires r4 for the reflection arm |
| C | Both resonators inserted | Requires r3 and r4; uses reflection phase for both arms |

## Key Parameters

| Symbol | Meaning |
|---|---|
| λ₀ | Operating wavelength |
| n | Effective refractive index |
| R1, R2 | Ring radii |
| r1, r2 | Main coupling coefficients |
| r3, r4 | Secondary coupling coefficients (needed for Config B/C) |
| Q1, Q2 | Quality factors |
| α1, α2 | Material attenuation (dB/cm) |
| η1, η2 | Phase shifters |

Coupling state is reported live in the status bar as **OC** (overcoupled, r < a) or **UC** (undercoupled, r > a).

## Author

Ahmad B. Yousafzai

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute, including for teaching and research.
