import sys, numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
a = np.load(sys.argv[1]).astype(float); s = a[0]-a[1]
print('row profile 0-60 (cols 120-480 mean):', np.round(s[:60,120:480].mean(1)).astype(int))
for nm, rr in [('top comb', slice(18, 50)), ('bottom comb', slice(470, 500))]:
    p = uniform_filter1d(s[rr].mean(0), 3)
    pk, _ = find_peaks(p, height=0.3*p.max(), distance=20)
    c = np.array([np.sum(np.arange(k-10,k+11)*p[k-10:k+11])/p[k-10:k+11].sum() for k in pk])
    print(nm, 'centroids', np.round(c,1), 'spacing', np.round(np.diff(c),1))
edges = [60,95,137,178,219,260,299,341,383,424,462]
for i in range(10):
    p = uniform_filter1d(s[edges[i]+3:edges[i+1]-3].mean(0), 2)
    g = np.gradient(p); le = 140+np.argmax(g[140:230]); te = 330+np.argmin(g[330:400])
    print(f'ch{i}: leading edge col {le}, trailing edge col {te}, span {te-le} px')
