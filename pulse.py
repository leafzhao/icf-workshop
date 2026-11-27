"""Helper functions for the OMEGA P510 pulse-shape workshop.

Most steps take a `Curves` object, return a new one, and draw a figure.
"""
from dataclasses import dataclass
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import rotate as rotate_image
from scipy.signal import find_peaks

FIDUCIAL_ROWS = {'top': slice(15, 50), 'bottom': slice(468, 505)}   # timing combs on the raw P510 image
CHANNEL_HEIGHT = 41                                                  # rows per beam channel on the raw image
BAND_COLORS = {10: '#DCE6F2', 5: '#A9C3E3', 1: '#6F97CC'}             # shading of the ±10 %, ±5 %, ±1 % bands

plt.rcParams.update({'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'legend.frameon': False, 'figure.constrained_layout.use': True})


@dataclass
class Curves:
    """A set of curves, one row per beam: P[i] is plotted against t[i]."""
    t: np.ndarray        # shape (n_beams, n_points)
    P: np.ndarray        # shape (n_beams, n_points)
    beams: list
    t_unit: str
    P_unit: str

    def plot(self, title='', beams='all'):
        """Plot the curves. `beams` is 'all', one beam number (12), or a list or range ([10, 25], range(20, 30))."""
        c = self if isinstance(beams, str) and beams == 'all' else select(self, beams)
        few = len(c.beams) <= 10
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for b, t, p in zip(c.beams, c.t, c.P):
            ax.plot(t, p, lw=1.2 if few else 0.7, label=f'beam {b}')
        ylabel = 'power (TW)' if c.P_unit == 'TW' else f'signal ({c.P_unit})'
        ax.set(xlabel=f'time ({c.t_unit})', ylabel=ylabel, title=f'{title}  [{len(c.beams)} beams]')
        if few:
            ax.legend(fontsize=8, ncols=2)
        plt.show()


def select(curves, beams):
    """Keep only the given beams (one number, or a list or range of numbers)."""
    beams = [beams] if np.isscalar(beams) else list(beams)
    missing = [b for b in beams if b not in curves.beams]
    if missing:
        raise ValueError(f'beams {missing} are not in this data; available: {curves.beams[0]} to {curves.beams[-1]}')
    rows = [curves.beams.index(b) for b in beams]
    return Curves(curves.t[rows], curves.P[rows], beams, curves.t_unit, curves.P_unit)


def integrate(t, P):
    """Time integral of each row of P (trapezoid rule)."""
    return (0.5 * (P[:, 1:] + P[:, :-1]) * np.diff(t, axis=1)).sum(axis=1)


def per_beam(beams, values, label, ax=None):
    """Scatter one number per beam, with the mean and shaded bands of ±1 %, ±5 %, ±10 % around it."""
    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(7, 3))
    m = values.mean()
    for pct, color in BAND_COLORS.items():
        ax.axhspan(m * (1 - pct / 100), m * (1 + pct / 100), color=color, lw=0, label=f'mean ±{pct}%')
    ax.axhline(m, color='k', lw=0.8, label='mean')
    ax.plot(beams, values, 'o', color='#004488', mec='k', mew=0.5, ms=5)
    ax.set(xlabel='beam number', ylabel=label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if show:
        ax.legend(fontsize=8, ncols=4, loc='lower center', bbox_to_anchor=(0.5, 1))
    inside = ', '.join(f'±{p}%: {np.sum(abs(values / m - 1) <= p / 100)}/{len(values)}' for p in sorted(BAND_COLORS))
    print(f'{label}: mean {m:.4g}, STDEV/mean {100 * values.std(ddof=1) / m:.1f} %; beams within {inside}')
    if show:
        plt.show()


def show_image(ax, image, cmap='gray'):
    """Raw image on a log colour scale, so the faint timing combs and the bright beams both show."""
    floor = 30   # about twice the background noise (~13 counts rms), so the empty areas stay dark
    return ax.imshow(image.clip(floor, None), cmap=cmap, norm=LogNorm(floor, np.percentile(image, 99.9)))


@dataclass
class RawImage:
    image: np.ndarray    # background-subtracted CCD counts, rows = space, columns = time
    cluster: int


# ---------- Step 1: import ----------
def load_raw(pattern):
    """Read every raw P510 file that matches `pattern` and subtract the background frame."""
    shots = []
    for path in sorted(glob(pattern)):
        with h5py.File(path, 'r') as f:
            frames = f['Streak_array'][...].astype(float)
            cluster = int(f.attrs['Unit'].decode().split('_')[1])
        shots.append(RawImage(frames[0] - frames[1], cluster))
        print(f'loaded {path}: cluster {cluster}, image {frames.shape[1]} x {frames.shape[2]} pixels')
    if not shots:
        raise FileNotFoundError(f'no file matches {pattern!r}; check the folder name')
    fig, axes = plt.subplots(1, len(shots), figsize=(5 * len(shots), 3.5), squeeze=False)
    for ax, s in zip(axes[0], shots):
        im = show_image(ax, s.image, cmap='viridis')
        ax.set(title=f'cluster {s.cluster} (log colour scale)', xlabel='column (pixel)  = time', ylabel='row (pixel)  = beam')
        fig.colorbar(im, ax=ax, label='counts')
    plt.show()
    return shots


# ---------- Step 2: raw curves ----------
def band_tilt(image, left=slice(200, 250), right=slice(310, 360)):
    """Mean tilt (degrees) of the 9 dark lines between beams, from their row in two column windows."""
    rows = []
    for cols in (left, right):
        profile = image[:, cols].sum(axis=1)
        # real gaps have prominence >= 0.44; dips inside a beam's own signal reach ~0.17 in a 50-column window
        dips, _ = find_peaks(-profile, prominence=0.3 * profile.max())
        assert len(dips) == 9, f'expected 9 dark lines between 10 beams, found {len(dips)}'
        y0, y1, y2 = profile[dips - 1], profile[dips], profile[dips + 1]
        rows.append(dips + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2))   # parabola through 3 points: sub-pixel minimum
    shift = np.mean(rows[1] - rows[0])
    return np.degrees(np.arctan(shift / (np.mean([right.start, right.stop]) - np.mean([left.start, left.stop]))))


def rotate(shots, angles):
    """Rotate each camera's image by `angles[cluster]` degrees, so the beam bands run horizontally."""
    out = [RawImage(rotate_image(s.image, angles.get(s.cluster, 0.0), reshape=False, order=1), s.cluster)
           for s in shots]
    zoom = (slice(100, 300), slice(150, 400))   # a few dark lines between beams, where a tilt is easiest to see
    fig, axes = plt.subplots(len(shots), 2, figsize=(11, 3.3 * len(shots)), squeeze=False)
    for (ax_before, ax_after), s, r in zip(axes, shots, out):
        print(f'cluster {s.cluster}: tilt of the bands {band_tilt(s.image):+.2f} deg, '
              f'rotate by {angles.get(s.cluster, 0.0):+.2f} deg -> tilt {band_tilt(r.image):+.2f} deg')
        for ax, img, when in [(ax_before, s.image, 'before'), (ax_after, r.image, 'after')]:
            show_image(ax, img[zoom])
            ax.set_aspect('auto')   # stretch the rows so a tilt of 1 degree is visible
            ax.set(title=f'cluster {s.cluster}, {when}: tilt {band_tilt(img):+.2f} deg', xlabel='column (pixel)',
                   ylabel='row (pixel)')
            dips = channel_edges(img)[1:-1] - zoom[0].start
            for y in dips[(dips > 0) & (dips < zoom[0].stop - zoom[0].start)]:
                ax.axhline(y, color='C3', lw=0.6, ls='--')
            ax.set_xticks(ax.get_xticks()[1:-1], [f'{x + zoom[1].start:.0f}' for x in ax.get_xticks()[1:-1]])
            ax.set_yticks(ax.get_yticks()[1:-1], [f'{y + zoom[0].start:.0f}' for y in ax.get_yticks()[1:-1]])
    plt.show()
    return out


def channel_edges(image):
    """Row boundaries of the 10 beam channels, found from the dark lines between them."""
    profile = image[:, 150:400].sum(axis=1)
    dips, _ = find_peaks(-profile, prominence=0.15 * profile.max())
    assert len(dips) == 9, f'expected 9 dark lines between 10 beams, found {len(dips)}'
    return np.r_[dips[0] - CHANNEL_HEIGHT, dips, dips[-1] + CHANNEL_HEIGHT]


def beam_rows(image):
    """(first row, last row + 1) summed for each of the 10 beams; skips the dark gaps at both edges of a channel."""
    e = channel_edges(image)
    return [(e[i] + 2, e[i + 1] - 1) for i in range(10)]


def extract_lineouts(shots):
    """Sum the rows of each beam channel: one raw curve (counts vs pixel) per beam."""
    P, beams = [], []
    for s in shots:
        for i, (lo, hi) in enumerate(beam_rows(s.image)):
            P.append(s.image[lo:hi].sum(axis=0))
            beams.append(10 * s.cluster + i)
    P = np.array(P)
    t = np.tile(np.arange(P.shape[1], dtype=float), (len(P), 1))
    raw = Curves(t, P, beams, 'pixel', 'counts')
    show_lineouts(raw, shots)
    return raw


def show_lineouts(raw, shots, beams='all'):
    """One row per camera: left, the image with the summed rows of each chosen beam boxed; right, their curves.

    `beams` is 'all', one beam number (12), or a list or range ([10, 25], range(20, 30)).
    """
    chosen = raw if isinstance(beams, str) and beams == 'all' else select(raw, beams)
    shots = [s for s in shots if any(b // 10 == s.cluster for b in chosen.beams)]
    fig, axes = plt.subplots(len(shots), 2, figsize=(12, 4.2 * len(shots)), squeeze=False, width_ratios=[1, 1.4])
    for (ax_img, ax_cur), s in zip(axes, shots):
        show_image(ax_img, s.image)
        for i, (lo, hi) in enumerate(beam_rows(s.image)):
            beam = 10 * s.cluster + i
            if beam not in chosen.beams:
                continue
            color = f'C{i}'
            ax_img.axhspan(lo - 0.5, hi - 0.5, fill=False, ec=color, lw=1)
            ax_img.text(5, (lo + hi) / 2, f'beam {beam}: rows {lo}-{hi - 1} ({hi - lo} rows)',
                        color=color, fontsize=7, va='center')
            ax_cur.plot(raw.t[raw.beams.index(beam)], raw.P[raw.beams.index(beam)], color=color, lw=1,
                        label=f'beam {beam}')
        ax_img.set(title=f'cluster {s.cluster}: rows summed for each beam', xlabel='column (pixel)', ylabel='row (pixel)')
        ax_cur.set(title=f'cluster {s.cluster}: raw curves', xlabel='column (pixel)', ylabel='signal (counts)')
        ax_cur.legend(fontsize=7, ncols=2)
    plt.show()


# ---------- Step 3: time calibration ----------
def fiducial_peaks(comb):
    """Pixel positions (centroids) of the pulses in a timing-comb lineout."""
    smooth = np.convolve(comb, np.ones(5) / 5, 'same')
    peaks, _ = find_peaks(smooth, height=0.3 * smooth.max(), distance=20)
    return np.array([np.average(np.arange(k - 10, k + 11), weights=comb[k - 10:k + 11]) for k in peaks])


def find_combs(shots):
    """Step 3a: find the comb pulses. Returns {(cluster, 'top' or 'bottom'): pulse pixel positions}."""
    combs = {}
    fig, axes = plt.subplots(len(shots), 2, figsize=(12, 3.6 * len(shots)), squeeze=False, width_ratios=[1, 1.4])
    for (ax_img, ax), s in zip(axes, shots):
        show_image(ax_img, s.image)
        for name, rows in FIDUCIAL_ROWS.items():
            color = 'C0' if name == 'top' else 'C1'
            comb = s.image[rows].mean(axis=0)
            peaks = fiducial_peaks(comb)
            assert len(peaks) == 8, f'cluster {s.cluster} {name} comb: expected 8 pulses, found {len(peaks)}'
            combs[s.cluster, name] = peaks
            ax_img.axhspan(rows.start - 0.5, rows.stop - 0.5, fill=False, ec=color, lw=1.2)
            ax_img.text(5, rows.stop + 4 if name == 'top' else rows.start - 4, f'{name} comb: rows {rows.start}-{rows.stop - 1}',
                        color=color, fontsize=8, va='top' if name == 'top' else 'bottom')
            ax.plot(comb, color=color, lw=0.8, label=f'{name} comb')
            ax.plot(peaks, np.interp(peaks, np.arange(comb.size), comb), 'v', color=color)
            print(f'cluster {s.cluster} {name} comb: pulses at pixel', np.round(peaks, 1))
        ax_img.set(title=f'cluster {s.cluster}: rows averaged for each comb', xlabel='column (pixel)', ylabel='row (pixel)')
        ax.set(xlabel='column (pixel)', ylabel='counts', title=f'cluster {s.cluster}: timing combs (triangles = found pulses)')
        ax.legend(fontsize=8)
    plt.show()
    return combs


def fit_time(combs, period_ps=548.7, degree=2):
    """Step 3b: fit time(pixel) with a polynomial of `degree`; neighbouring comb pulses are `period_ps` apart."""
    fits = {}
    clusters = sorted({c for c, _ in combs})
    fig, axes = plt.subplots(len(clusters), 3, figsize=(11, 3 * len(clusters)), sharex=True, squeeze=False)
    for (cluster, name), peaks in combs.items():
        ax = axes[clusters.index(cluster)]
        color = 'C0' if name == 'top' else 'C1'      # same colours as the comb figure of step 3a
        known_t = np.arange(len(peaks)) * period_ps
        coef = np.polyfit(peaks, known_t, degree)
        residual = np.polyval(coef, peaks) - known_t
        fits[cluster, name] = coef
        px = np.linspace(peaks[0], peaks[-1], 200)
        ax[0].plot(peaks, known_t, 'o', ms=4, color=color, label=f'{name} comb')
        ax[0].plot(px, np.polyval(coef, px), color=color, lw=1)
        ax[1].plot(px, np.polyval(np.polyder(coef), px), color=color, lw=1, label=f'{name} comb')
        ax[2].plot(peaks, residual, 'o-', ms=4, color=color, label=f'{name}: rms {residual.std():.1f} ps')
        print(f'cluster {cluster} {name}: fit residual rms {residual.std():.1f} ps')
    for cluster, ax in zip(clusters, axes):
        ax[0].set(ylabel='time (ps)', title=f'cluster {cluster}: comb pulses and fit')
        ax[1].set(ylabel='sweep (ps / pixel)', title=f'cluster {cluster}: slope of the fit')
        ax[2].set(ylabel='fit residual (ps)', title=f'cluster {cluster}: residual (degree {degree})')
        for a in ax:
            a.legend(fontsize=7)
    for a in axes[-1]:
        a.set_xlabel('column (pixel)')
    plt.show()
    return fits


def apply_time(raw, shots, fits):
    """Step 3c: give every beam its own time axis, a mix of the top and bottom fits weighted by its row."""
    t = np.empty_like(raw.t)
    px = raw.t[0]
    top, bottom = (np.mean([r.start, r.stop]) for r in FIDUCIAL_ROWS.values())
    fig, axes = plt.subplots(len(shots), 1, figsize=(7, 3 * len(shots)), sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], shots):
        # the two combs have different delays; only the shape of t(pixel) matters before alignment
        curve = {name: np.polyval(fits[s.cluster, name], px) - np.polyval(fits[s.cluster, name], px[0])
                 for name in FIDUCIAL_ROWS}
        edges = channel_edges(s.image)
        w = (0.5 * (edges[:-1] + edges[1:]) - top) / (bottom - top)
        for i in range(10):
            row = raw.beams.index(10 * s.cluster + i)
            t[row] = (1 - w[i]) * curve['top'] + w[i] * curve['bottom']
            ax.plot(t[row], raw.P[row], color=f'C{i}', lw=1, label=f'beam {10 * s.cluster + i}')
        print(f'cluster {s.cluster}, weight of the bottom comb for beams {10 * s.cluster}-{10 * s.cluster + 9}:',
              np.round(w, 2))
        ax.set(ylabel=f'signal ({raw.P_unit})', title=f'cluster {s.cluster}: time axis calibrated')
        ax.legend(fontsize=7, ncols=2)
    axes[-1, 0].set_xlabel('time (ps)')
    plt.show()
    return Curves(t, raw.P, raw.beams, 'ps', raw.P_unit)


# ---------- Step 4: power calibration ----------
def load_beam_energy(path):
    """UV energy on target of every beam (J), measured by the calorimeters."""
    with h5py.File(path, 'r') as f:
        return {int(b): float(e) for c in range(1, 7)
                for b, e in zip(f[f'DATA_CL{c}'].attrs['BEAMS'], f[f'DATA_CL{c}'].attrs['TARGET_UV'])}


# ---------- Step 5: time alignment ----------
def edge_time(t, p, edge, level):
    """Time of a feature of one curve: 'rise' or 'fall' through `level` x peak, or 'peak'."""
    if edge == 'peak':
        return t[np.argmax(p)]
    if edge not in ('rise', 'fall'):
        raise ValueError(f"edge must be 'rise', 'fall' or 'peak', not {edge!r}")
    y = level * p.max()
    above = np.flatnonzero(p >= y)
    i = above[0] if edge == 'rise' else above[-1]
    k = [i - 1, i] if edge == 'rise' else [i + 1, i]   # np.interp needs the two points in increasing p
    return np.interp(y, p[k], t[k])


def align(curves, edge='rise', level=0.05, dt=10.0):
    """Shift each curve so its `edge` feature sits at t = 0, then put all curves on one time grid of step `dt`."""
    t = np.array([ti - edge_time(ti, p, edge, level) for ti, p in zip(curves.t, curves.P)])
    grid = np.arange(np.ceil(t[:, 0].max() / dt) * dt, t[:, -1].min(), dt)   # the range every beam covers
    P = np.array([np.interp(grid, ti, p) for ti, p in zip(t, curves.P)])
    # drop the empty baseline: keep the pulse (mean above 1 % of its peak) plus 300 ps on each side
    on = np.flatnonzero(P.mean(axis=0) > 0.01 * P.mean(axis=0).max())
    keep = slice(max(on[0] - int(300 / dt), 0), on[-1] + int(300 / dt) + 1)
    grid, P = grid[keep], P[:, keep]
    out = Curves(np.tile(grid, (len(P), 1)), P, curves.beams, curves.t_unit, curves.P_unit)
    out.plot(f'Step 5: aligned, t = 0 at the {edge}' + ('' if edge == 'peak' else f' through {level:.0%} of peak'))
    return out


# ---------- Step 6: average ----------
def average(curves):
    """Mean and standard deviation over beams, at each time point."""
    mean, std = curves.P.mean(axis=0), curves.P.std(axis=0, ddof=1)
    t = curves.t[0]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(curves.t.T, curves.P.T, color='0.75', lw=0.5)
    ax.fill_between(t, mean - std, mean + std, color='tab:blue', alpha=0.3, label='mean ± 1 STDEV')
    ax.plot(t, mean, color='tab:blue', lw=2, label='mean')
    ax.set(xlabel=f'time ({curves.t_unit})', ylabel=f'power per beam ({curves.P_unit})',
           title=f'Step 6: average of {len(curves.beams)} beams')
    ax.legend(); plt.show()
    return Curves(t[None], mean[None], ['mean'], curves.t_unit, curves.P_unit)


# ---------- Step 7: statistics ----------
def statistics(curves):
    """Energy and peak power of every beam, and how far each beam's power is from the mean at each time."""
    t = curves.t[0]
    energy = integrate(curves.t, curves.P)
    peak = curves.P.max(axis=1)
    t_peak = t[curves.P.argmax(axis=1)]

    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    per_beam(curves.beams, energy, 'energy (J)', ax=ax[0])
    per_beam(curves.beams, peak, 'peak power (TW)', ax=ax[1])
    ax[0].set_xlabel('')
    ax[0].legend(fontsize=8, ncols=4, loc='lower center', bbox_to_anchor=(0.5, 1))
    fig.suptitle(f'{len(curves.beams)} beams: each dot is one beam')
    plt.show()
    # a relative spread of a time is meaningless: t = 0 was chosen by us in step 5
    print(f'time of peak (ps): mean {t_peak.mean():.0f}, STDEV {t_peak.std(ddof=1):.0f}')

    mean, std = curves.P.mean(axis=0), curves.P.std(axis=0, ddof=1)
    main = mean > 0.1 * mean.max()
    balance = std[main] / mean[main]
    print(f'power balance (STDEV/mean over time, where mean > 10% of peak): {100 * balance.mean():.1f} %')
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True, height_ratios=[2, 1])
    ax[0].plot(t, curves.P.T, color='0.6', lw=0.5)
    for pct in (5, 1):
        ax[0].fill_between(t, mean * (1 - pct / 100), mean * (1 + pct / 100), color=BAND_COLORS[pct],
                           lw=0, label=f'mean ±{pct}%')
    ax[0].plot(t, mean, 'k', lw=1, label='mean')
    ax[0].set(ylabel=f'power per beam ({curves.P_unit})', title='grey: single beams')
    ax[0].legend()
    ax[1].plot(t, np.where(main, 100 * std / mean, np.nan), color='#004488')   # NaN breaks the line outside the pulse
    for pct in (1, 5):
        ax[1].axhline(pct, color='0.4', lw=0.8, ls='--')
        ax[1].text(t[0], pct, f' {pct}%', va='bottom', fontsize=8, color='0.4')
    # the rising and falling edges reach tens of %; cut them off so the 1 % and 5 % lines stay readable
    ax[1].set(xlabel=f'time ({curves.t_unit})', ylabel='STDEV/mean (%)', ylim=(0, 15),
              title='spread between beams, where mean > 10% of peak')
    plt.show()
    return {'energy_J': energy, 'peak_TW': peak, 't_peak_ps': t_peak, 'power_balance': balance.mean()}


# ---------- Part 2: the facility's 60-beam product ----------
def load_official(path):
    """The facility's own 60-beam result (already calibrated), as Curves."""
    with h5py.File(path, 'r') as f:
        t, P = f['SUMMARY_DATA_TIME'][...], f['SUMMARY_DATA_UV'][...].T
        beams = [int(b) for b in f['SUMMARY_DATA_UV'].attrs['BEAMS']]
    curves = Curves(np.tile(t, (len(P), 1)), P, beams, 'ps', 'TW')
    curves.plot('All 60 beams, calibrated by the facility')
    return curves
