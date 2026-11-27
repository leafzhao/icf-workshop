"""Quick-look of OMEGA P510 processed UV pulse shapes (p510_data_<shot>.h5)."""
import sys, numpy as np, h5py, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.2, 'legend.frameon': False, 'figure.constrained_layout.use': True, 'pdf.fonttype': 42,
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']})
src, out = sys.argv[1], sys.argv[2]
f = h5py.File(src, 'r')
t = f['SUMMARY_DATA_TIME'][...]                      # ps
uv = f['SUMMARY_DATA_UV'][...]                       # TW per beam, (512, 60)
avg = f['SUMMARY_DATA_AVERAGE'][:, 0]
beams = f['SUMMARY_DATA_UV'].attrs['BEAMS']; bt0 = f['SUMMARY_DATA_UV'].attrs['BEAM_T0']
dt = np.diff(t).mean()
E = uv.sum(0) * dt                                   # TW*ps = J
tot = uv.sum(1)
def fwhm(y):
    i = np.where(y >= y.max() / 2)[0]; return t[i[-1]] - t[i[0]]
def edges(y, lvl):
    i = np.where(y >= lvl * y.max())[0]; return t[i[0]], t[i[-1]]
cl = {c: f[f'DATA_CL{c}'].attrs for c in range(1, 7)}
tgt = np.concatenate([cl[c]['TARGET_UV'] for c in cl])
print(f"shot {f.attrs['LOG_NUM']}  {f.attrs['SHOT_TIME'].decode()}  pulse shape {f['SUMMARY_DATA_AVERAGE'].attrs['SHAPE'].decode()}")
print(f"time axis {t[0]:.0f} .. {t[-1]:.0f} ps, dt {dt:.2f} ps; fiducial peaks (ps) {np.round(f['SUMMARY_DATA_FIDU'].attrs['PKS'],1)}")
print(f"fiducial period {np.diff(f['SUMMARY_DATA_FIDU'].attrs['PKS']).mean():.1f} ps")
print(f"total peak power {tot.max():.2f} TW at {t[tot.argmax()]:.0f} ps; 2%-98% window {edges(tot,0.02)} ps; FWHM {fwhm(tot):.0f} ps")
print(f"total UV energy (integral of P510) {E.sum():.0f} J; sum TARGET_UV attrs {tgt.sum():.0f} J")
print(f"per-beam energy (P510 integral): mean {E.mean():.1f} J, std {E.std():.1f} J ({100*E.std()/E.mean():.1f}% rms), min {E.min():.1f} (beam {beams[E.argmin()]}), max {E.max():.1f} (beam {beams[E.argmax()]})")
print(f"per-beam TARGET_UV attr: mean {tgt.mean():.1f} J, {100*tgt.std()/tgt.mean():.1f}% rms")
print(f"BEAM_T0 (2% rise): mean {bt0.mean():.1f} ps, rms {bt0.std():.1f} ps, range {bt0.min():.1f} .. {bt0.max():.1f}")
print(f"power balance AVG_PB {f['SUMMARY_DATA_PB_0'].attrs['AVG_PB']:.3f}, PEAK {f['SUMMARY_DATA_PB_0'].attrs['PEAK']:.2f}")
for c in cl: print(f"  CL{c}: beams {cl[c]['BEAMS'][0]}-{cl[c]['BEAMS'][-1]}, TARGET_UV mean {cl[c]['TARGET_UV'].mean():.1f} J, TIME_PIX {cl[c]['TIME_PIX']:.2f} ps, warn: {cl[c]['WARNING'].decode().strip() or '-'}")
# shape landmarks of the average pulse
pk = avg.max(); print(f"average pulse: peak {pk:.3f} TW/beam; level at t=0,500,1000,1500,2000 ps:", [round(float(np.interp(x, t, avg)) / pk, 3) for x in (0, 500, 1000, 1500, 2000)])
assert abs(E.sum() - tgt.sum()) / tgt.sum() < 0.2, 'integrated P510 energy far from TARGET_UV'

fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
m = (t > -500) & (t < 3500)
ax[0, 0].plot(t[m], uv[m], c='#BBBBBB', lw=0.5); ax[0, 0].plot(t[m], avg[m], c='#004488', lw=1.8, label='average')
ax[0, 0].set(xlabel='Time $t$ (ps)', ylabel='UV power per beam (TW)'); ax[0, 0].legend(loc='upper left', bbox_to_anchor=(0.08, 1))
ax[0, 1].plot(t[m], tot[m], c='#BB5566'); ax[0, 1].set(xlabel='Time $t$ (ps)', ylabel='Total UV power, 60 beams (TW)')
ax[1, 0].bar(beams.astype(str), E, color=[plt.cm.viridis(i / 6) for i in np.repeat(np.arange(6), 10)], width=0.8)
ax[1, 0].axhline(E.mean(), c='gray', ls='--', lw=0.8); ax[1, 0].set(xlabel='Beam', ylabel='UV energy (J)', ylim=(E.min() * 0.9, E.max() * 1.05))
ax[1, 0].set_xticks(np.arange(0, 60, 10), beams[::10].astype(str))
ax[1, 1].plot(np.arange(60), bt0, 'o', mfc='#004488', mec='k', ms=4, mew=0.4)
ax[1, 1].set(xlabel='Beam', ylabel='2% rise time $t_0$ (ps)'); ax[1, 1].set_xticks(np.arange(0, 60, 10), beams[::10].astype(str))
for a, l in zip(ax.flat, 'abcd'): a.text(0.02, 0.97, f'({l})', transform=a.transAxes, va='top', fontweight='bold', fontsize=9)
fig.savefig(out, dpi=300)
