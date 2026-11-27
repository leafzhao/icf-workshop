"""Helper functions for the time-resolved scattered-light spectrum (tSPM) workshop.

Every step takes a `Spectrogram` and returns a new one, and draws a figure.
"""
from dataclasses import dataclass, replace
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from scipy import ndimage

C_M_PER_NS = 0.299792458
# Sellmeier coefficients of fused silica (Malitson 1965), wavelength in um
SELLMEIER_B = (0.6961663, 0.4079426, 0.8974794)
SELLMEIER_C = (0.0684043**2, 0.1162414**2, 9.896161**2)

# the white-to-red map used in the original analysis notebook
WHITE_JET = LinearSegmentedColormap.from_list('white_jet', list(zip(
    np.linspace(0, 1, 10),
    ['w', '#234b9f', '#2a66b1', '#4bbfe1', '#7bc59a', '#b4d351', '#f5cc1d', '#ec6319', '#df1814', '#7a171c'])))


@dataclass
class Spectrogram:
    """Intensity I[i, j] at wavelength lam[i] and time t[j]."""
    t: np.ndarray        # time (ns), shape (n_t,)
    lam: np.ndarray      # wavelength (nm), shape (n_lam,), in the order stored in the file
    I: np.ndarray        # counts, shape (n_lam, n_t)


def pixel_edges(x):
    """Pixel boundaries from pixel centres (n + 1 values for n centres)."""
    mid = (x[1:] + x[:-1]) / 2
    return np.r_[2 * x[0] - mid[0], mid, 2 * x[-1] - mid[-1]]


def plot_map(s, ax, log=True, cmap='viridis', vmin=100, vmax=3e4, xlim=None, ylim=None):
    """Draw a spectrogram on `ax` with time on x and wavelength on y; returns the image."""
    norm = LogNorm(vmin, vmax) if log else Normalize(0, vmax)
    order = np.argsort(s.lam)
    # pcolorfast, not imshow: the streak sweep makes the time steps unequal
    # zeros would be blank on a log scale; NaN (no data, from the GVD shift) stays blank
    I = np.clip(s.I[order], vmin, None) if log else s.I[order]
    im = ax.pcolorfast(pixel_edges(s.t), pixel_edges(s.lam[order]), I, cmap=cmap, norm=norm)
    ax.set(xlabel='time (ns)', ylabel='wavelength (nm)', xlim=xlim, ylim=ylim)
    plt.colorbar(im, ax=ax, label='counts')
    return im


def plot_maps(s, panels, **kw):
    """One panel per dict in `panels` (keys: title, log, cmap, vmin, vmax); `kw` go to every panel."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.3), layout='constrained', squeeze=False)
    for ax, p in zip(axes[0], panels):
        p = dict(p)
        ax.set_title(p.pop('title', ''))
        plot_map(s, ax, **{**kw, **p})
    plt.show()


# ---------- Step 1: raw data ----------
def load_dac(path):
    """Read a .dac file: row 0 holds the wavelengths, column 0 the times, the rest the counts.

    The file stores rows = time, columns = wavelength; we transpose so that time runs along x.
    """
    with open(path) as f:
        lam = np.array(f.readline().split()[1:], dtype=float)
    data = np.loadtxt(path, skiprows=1)
    t, counts = data[:, 0], data[:, 1:].T
    assert counts.shape == (len(lam), len(t)), 'axes and image size disagree'
    s = Spectrogram(t, lam, counts)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    im = ax[0].imshow(counts, aspect='auto', cmap='viridis', norm=LogNorm(100, 3e4))
    ax[0].set(title='raw image, log scale', xlabel='column (pixel)  = time', ylabel='row (pixel)  = wavelength')
    fig.colorbar(im, ax=ax[0], label='counts')
    ax[1].hist(counts.ravel(), bins=np.arange(counts.min(), 600, 2), log=True, color='0.4')
    ax[1].set(title='histogram of all pixels', xlabel='counts', ylabel='number of pixels')
    plt.show()
    print(f'loaded {path}: {len(lam)} wavelengths x {len(t)} times, counts {counts.min():.0f} to {counts.max():.0f}')
    return s


# ---------- Step 2: axes from the .dac file ----------
def show_axes(s):
    """The axes stored in the .dac file: what each pixel means, and the map in physical units."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3), layout='constrained')
    ax[0].plot(1e3 * np.diff(s.t), color='k')
    ax[0].set(xlabel='column (pixel)', ylabel='time step (ps / pixel)', title='streak sweep: time per pixel')
    ax[1].plot(s.lam, color='k')
    ax[1].set(xlabel='row (pixel)', ylabel='wavelength (nm)', title=f'wavelength: {np.diff(s.lam).mean():.4f} nm / pixel')
    ax[2].set_title('raw map, physical axes')
    plot_map(s, ax[2])
    plt.show()
    print(f'time {s.t[0]:.3f} to {s.t[-1]:.3f} ns, {1e3 * np.diff(s.t).min():.2f} to {1e3 * np.diff(s.t).max():.2f} ps / pixel')
    print(f'wavelength {s.lam[0]:.2f} to {s.lam[-1]:.2f} nm, {abs(np.diff(s.lam).mean()):.4f} nm / pixel')


# ---------- Step 3: preprocessing ----------
def preprocess(s, t_dark, median=3, sigma=2, **plot_kw):
    """Subtract the background, then remove spikes and smooth.

    t_dark: times (ns) before this contain no signal; their median is the background.
    """
    background = np.median(s.I[:, s.t < t_dark])
    I = np.clip(s.I - background, 0, None)
    despiked = ndimage.median_filter(I, size=median)
    smooth = ndimage.gaussian_filter(despiked, sigma)
    print(f'background = median of t < {t_dark} ns = {background:.1f} counts')

    steps = [('before smoothing', I), (f'+ median filter {median}x{median}', despiked),
             (f'+ Gaussian sigma = {sigma} px', smooth)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), layout='constrained')
    for ax, (title, img) in zip(axes, steps):
        ax.set_title(title)
        plot_map(replace(s, I=img), ax, **plot_kw)
    plt.show()
    return replace(s, I=smooth)


# ---------- Step 4: group-velocity dispersion ----------
def group_index(lam_nm):
    """Group index n - lam dn/dlam of fused silica."""
    lam = lam_nm / 1e3
    n = np.sqrt(1 + sum(b * lam**2 / (lam**2 - c) for b, c in zip(SELLMEIER_B, SELLMEIER_C)))
    return n - lam * np.gradient(n, lam)


def gvd_delay(lam, fiber_m):
    """Arrival delay (ns) of each wavelength after `fiber_m` of fused silica, relative to the shortest wavelength.

    Longer wavelengths travel faster, so their delay is negative: they arrive early.
    """
    ref = lam.argmin()
    Ng = group_index(lam)
    delay = fiber_m / C_M_PER_NS * (Ng - Ng[ref])   # ns, <= 0

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), layout='constrained')
    ax[0].plot(lam, Ng, color='k')
    ax[0].set(xlabel='wavelength (nm)', ylabel='group index $N_g$', title='fused silica (Sellmeier)')
    ax[1].plot(lam, delay, color='k')
    ax[1].set(xlabel='wavelength (nm)', ylabel='delay (ns)', title=f'{fiber_m} m fiber, ref. {lam[ref]:.0f} nm')
    plt.show()
    print(f'delay across the window: {np.ptp(delay):.3f} ns')
    return delay


def gvd_correct(s, delay, **plot_kw):
    """Shift each wavelength row later by its fiber delay: corrected(t) = measured(t + delay)."""
    # sub-pixel shift by linear interpolation; times outside the record become NaN
    I = np.array([np.interp(s.t + d, s.t, row, left=np.nan, right=np.nan) for d, row in zip(delay, s.I)])
    out = replace(s, I=I)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), layout='constrained')
    for a, spec, title in [(ax[0], s, 'before'), (ax[1], out, 'after GVD correction')]:
        a.set_title(title)
        plot_map(spec, a, **plot_kw)
    plt.show()
    print(f'largest shift: {np.ptp(delay) / np.diff(s.t).mean():.0f} pixels')
    return out


# ---------- Step 7: laser pulse ----------
def load_laser(pattern, pulse_ns):
    """Read every oscilloscope trace matching `pattern`; return (t, P) with each beam normalised to its peak.

    Two oscilloscope formats occur: Keysight (header, then t,V from line 23)
    and Tektronix (t,V in columns 4 and 5). t = 0 is put `pulse_ns` before
    the falling edge (last point above peak/e), so the pulse starts near 0.
    """
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f'no file matches {pattern!r}; check the folder name')
    grid = np.arange(-2, 8, 0.005)
    P = []
    for f in files:
        with open(f) as fh:
            tek = fh.readline().startswith('"Record Length"')
        tv = np.genfromtxt(f, delimiter=',', usecols=(3, 4)) if tek else np.loadtxt(f, delimiter=',', skiprows=22)
        t, p = tv[:, 0] * 1e9, tv[:, 1]
        p = p - np.median(p[:200])                       # oscilloscope offset, from the samples before the pulse
        t = t - t[np.where(p > p.max() / np.e)[0][-1]] + pulse_ns
        P.append(np.interp(grid, t, p / p.max()))
    P = np.array(P)

    fig, ax = plt.subplots(figsize=(7, 3.3))
    for f, p in zip(files, P):
        ax.plot(grid, p, lw=0.7, label='B' + f.rsplit('_', 1)[1].removesuffix('.csv'))
    ax.plot(grid, P.mean(axis=0), 'k', lw=2, label='mean')
    ax.set(xlabel='time (ns)', ylabel='power (normalised)', title=f'{len(files)} beams, each normalised to its peak')
    ax.legend(ncol=3, fontsize=7, frameon=False)
    plt.show()
    return grid, P


def overlay_laser(s, t_laser, p_laser, band, color='w', **plot_kw):
    """Draw the spectrum with the laser pulse on top, scaled into the wavelength range `band` (nm)."""
    fig, ax = plt.subplots(figsize=(5.5, 4), layout='constrained')
    plot_map(s, ax, **plot_kw)
    ax.plot(t_laser, band[0] + (band[1] - band[0]) * p_laser / p_laser.max(), color=color, lw=1)
    plt.show()


# ---------- Step 8: lineouts ----------
def lineouts(s, bands, t_laser=None, p_laser=None, xlim=None, ylim=None):
    """Time history of each wavelength band {label: (lam_min, lam_max)}, each normalised to its peak.

    Left: the map with a box around each band. Right: the lineouts, in the same colours.
    """
    colors = ['#EE6677', '#CCBB44', '#66CCEE', '#AA3377', '#228833']   # Tol bright: readable on the dark map
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), layout='constrained')
    plot_map(s, ax[0], xlim=xlim, ylim=ylim)
    ax[0].set_title('preview')
    t0, t1 = xlim if xlim else (s.t[0], s.t[-1])
    if t_laser is not None:
        ax[1].fill_between(t_laser, p_laser / p_laser.max(), color='0.85', label='laser')
    for c, (label, (l0, l1)) in zip(colors, bands.items()):
        ax[0].add_patch(plt.Rectangle((t0, l0), t1 - t0, l1 - l0, fill=False, ec=c, lw=1.2))
        curve = np.nanmean(s.I[(s.lam >= l0) & (s.lam <= l1)], axis=0)
        ax[1].plot(s.t, curve / np.nanmax(curve), color=c, lw=1.2, label=f'{label} ({l0}-{l1} nm)')
    ax[1].set(xlabel='time (ns)', ylabel='intensity (normalised)', xlim=xlim, ylim=(0, 1.1), title='lineouts')
    ax[1].legend(frameon=False, fontsize=8)
    plt.show()
