import sys, numpy as np, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
mpl.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'axes.linewidth':0.6,'lines.linewidth':1.2,'legend.frameon':False,'figure.constrained_layout.use':True,'font.sans-serif':['Arial','Helvetica','DejaVu Sans']})
a = np.load(sys.argv[1]).astype(float)
sig = a[0] - a[1]
fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))
for i, (img, lab) in enumerate([(a[0], 'frame 0 (raw)'), (a[1], 'frame 1 (raw)')]):
    im = ax[0, i].imshow(img, origin='upper', cmap='viridis', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.9))
    ax[0, i].set(title=lab, xlabel='column (px)', ylabel='row (px)'); fig.colorbar(im, ax=ax[0, i], shrink=.8, label='counts')
im = ax[0, 2].imshow(np.clip(sig, 1, None), origin='upper', cmap='viridis', norm=LogNorm(vmin=10, vmax=sig.max()))
ax[0, 2].set(title='frame0 - frame1 (log)', xlabel='column (px)', ylabel='row (px)'); fig.colorbar(im, ax=ax[0, 2], shrink=.8, label='counts')
ax[1, 0].plot(sig.sum(0), c='#004488'); ax[1, 0].set(xlabel='column (px)', ylabel='sum over rows (counts)')
ax[1, 1].plot(sig.sum(1), c='#BB5566'); ax[1, 1].set(xlabel='row (px)', ylabel='sum over columns (counts)')
ax[1, 2].hist(a[1].ravel(), bins=200, color='gray', log=True); ax[1, 2].set(xlabel='frame 1 counts', ylabel='pixels')
fig.savefig(sys.argv[2], dpi=150)
# numbers
r, c = np.unravel_index(sig.argmax(), sig.shape)
print('peak', sig.max(), 'at row', r, 'col', c)
for axis, nm in [(0, 'column profile'), (1, 'row profile')]:
    p = sig.sum(axis); p = p - np.median(p)
    half = np.where(p > p.max()/2)[0]
    print(nm, 'peak idx', p.argmax(), 'FWHM-span', half.min(), '-', half.max(), 'n=', half.size)
m = sig > 0.1*sig.max()
rows, cols = np.where(m); print('>10% region rows', rows.min(), rows.max(), 'cols', cols.min(), cols.max())
print('bg-sub total counts', sig.sum(), ' frame1 row-mean range', a[1].mean(1).min(), a[1].mean(1).max(), ' col-mean range', a[1].mean(0).min(), a[1].mean(0).max())
print('frame0 corner (bg) median', np.median(a[0][:50,:50]), np.median(a[0][-50:,-50:]))
