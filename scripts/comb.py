import sys, numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
a = np.load(sys.argv[1]).astype(float); s = a[0]-a[1]
for nm, rr in [('top comb', slice(38, 56)), ('bottom comb', slice(470, 500))]:
    p = s[rr].mean(0); p = uniform_filter1d(p, 3)
    pk, _ = find_peaks(p, height=0.3*p.max(), distance=10)
    # centroid refine
    c = [np.sum(np.arange(k-8,k+9)*p[k-8:k+9])/p[k-8:k+9].sum() for k in pk]
    print(nm, 'peaks', np.round(c,1), 'spacing', np.round(np.diff(c),2), 'mean', np.diff(c).mean().round(2), 'heights', np.round(p[pk]).astype(int))
rp = s[:, 150:400].sum(1)
dips, _ = find_peaks(-rp, prominence=0.15*rp.max())
print('channel separator rows', dips, 'spacing', np.diff(dips))
edges = np.r_[60, dips, 462]
print('ch  rows      onset50%  cutoff50%  dur(px)  peak  sum')
for i in range(len(edges)-1):
    r0, r1 = edges[i]+3, edges[i+1]-3
    if r1 <= r0: continue
    p = uniform_filter1d(s[r0:r1].mean(0), 3); m = p[150:420].max()
    above = np.where(p > 0.5*m)[0]
    print(f'{i:2d} {r0:3d}-{r1:3d}  {above.min():6d}  {above.max():8d}  {above.max()-above.min():6d}  {m:7.0f}  {s[r0:r1].sum():.3g}')
