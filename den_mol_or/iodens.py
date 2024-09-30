import numpy as np
import os
import struct
from numba import njit

OUT = "DENS/"

def uread(fname):
    f = open(fname, "rb")
    lf = f.read()
    f.close()
    n = struct.unpack("i", lf[4:8])[0]
    ans = []
    for i in range(n):
        tmp = []
        for j in range(9):
            tmp.append(struct.unpack("d", lf[80 * i + 20 + 8 * j: 80 * i + 28 + 8 * j])[0])
        ans.append(tmp)
    return np.array(ans)


def ureadu(fname):
    f = open(fname, "rb")
    lf = f.read()
    f.close()
    n = struct.unpack("i", lf[4:8])[0]
    ans = []
    for i in range(n):
        tmp = []
        for j in range(14):
            tmp.append(struct.unpack("d", lf[8 * 15 * i + 20 + 8 * j: 8 * 15 * i + 28 + 8 * j])[0])
        ans.append(tmp)
    return np.array(ans)


def read_mwfn(dire):
    return np.vstack([np.loadtxt(dire + 'rho', skiprows=1)[:,3], np.loadtxt(dire + 'grad', skiprows=1)[:,3], np.loadtxt(dire + 'lapl', skiprows=1)[:,3]]).T


def gennpz_mwfn(dire):
    dire += "/"
    keys = os.listdir(dire)
    dfile = dict.fromkeys(keys)
    for key in keys:
        dfile[key] = read_mwfn(dire + key + "/")
    np.savez_compressed(OUT + dire[:-1].split("/")[1] + ".npz", **dfile)


def gennpz(dire):
    dire += "/"
    keys = os.listdir(dire)
    dfile = dict.fromkeys(keys)
    for key in keys:
        dfile[key] = uread(dire + key + "/DENSITY")
    np.savez_compressed(OUT + dire[:-1] + ".npz", **dfile)


def gennpzu(dire):
    dire += "/"
    keys = os.listdir(dire)
    dfile = dict.fromkeys(keys)
    for key in keys:
        dfile[key] = ureadu(dire + key + "/DENSITY")
    np.savez_compressed(OUT + dire[:-1] + ".npz", **dfile)


@njit
def jsum(x):
    ans = 0
    for i in x:
        ans += i
    return ans


def niad_mwfn(density, densityr):
    w    = densityr[:,3]

    rho  = density[:,0]
    grd  = density[:,1]
    lr   = density[:,2]
    elnum = jsum(rho * w)

    rhor  = densityr[:,4]
    grdxr = densityr[:,5]
    grdyr = densityr[:,6]
    grdzr = densityr[:,7]
    grdr  = np.sqrt(grdxr ** 2 + grdyr ** 2 + grdzr ** 2)
    lrr   = densityr[:,8]

    lniad = []
    lniad.append(jsum(w * abs(rhor - rho)) / elnum)
    lniad.append(jsum(w * abs(grdr - grd)) / elnum)
    lniad.append(jsum(w * abs(lrr - lr )) / elnum)

    return lniad


def niad(density, densityr):
    w    = density[:,3]
    rho  = density[:,4]
    grdx = density[:,5]
    grdy = density[:,6]
    grdz = density[:,7]
    grd  = np.sqrt(grdx ** 2 + grdy ** 2 + grdz ** 2)
    lr   = density[:,8]
    elnum = jsum(rho * w)

    rhor  = densityr[:,4]
    grdxr = densityr[:,5]
    grdyr = densityr[:,6]
    grdzr = densityr[:,7]
    grdr  = np.sqrt(grdxr ** 2 + grdyr ** 2 + grdzr ** 2)
    lrr   = densityr[:,8]

    lniad = []
    lniad.append(jsum(w * abs(rhor - rho)) / elnum)
    lniad.append(jsum(w * abs(grdr - grd)) / elnum)
    lniad.append(jsum(w * abs(lrr - lr )) / elnum)

    return lniad
