import sys, numpy as np, h5py
from pyhdf.SD import SD, SDC
h4 = SD(sys.argv[2], SDC.READ)
print("HDF4 global attrs:", h4.attributes())
print("HDF4 datasets:", h4.datasets())
f = h5py.File(sys.argv[1], 'r')
a5 = f['Streak_array'][...]
for name in h4.datasets():
    s = h4.select(name)
    a4 = s.get()
    print(name, a4.shape, a4.dtype, "dims:", s.dimensions())
    at4 = s.attributes()
    at5 = dict(f['Streak_array'].attrs)
    if name == 'Streak_array':
        print("arrays equal:", np.array_equal(a4, a5))
        only4 = set(at4) - set(at5); only5 = set(at5) - set(at4)
        print("attrs only in hdf4:", only4, " only in h5:", only5)
        diff = [k for k in set(at4) & set(at5) if str(np.asarray(at4[k]).tolist()).encode() != str(np.asarray(at5[k]).tolist()).encode() and not np.array_equal(np.atleast_1d(at4[k]).astype(str), np.atleast_1d(at5[k]).astype(str))]
        print("attrs differing:", [(k, at4[k], at5[k]) for k in diff])
for i in range(a5.shape[0]):
    x = a5[i].astype(float)
    print(f"frame {i}: min {x.min():.0f} max {x.max():.0f} mean {x.mean():.1f} median {np.median(x):.0f} std {x.std():.1f}  n_sat(65535) {(a5[i]==65535).sum()}  n>=4095 {(a5[i]>=4095).sum()}")
d = a5[0].astype(float) - a5[1].astype(float)
print("frame0-frame1: min", d.min(), "max", d.max(), "mean", d.mean())
np.save(sys.argv[3], a5)
