#!/usr/bin/env python

import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from mytools import (masked_interp, find_nearest, smooth1d)
from scipy.interpolate import interp1d
from pycurrents.system import Bunch
# from pycurrents.file import npzfile


def round_of_rating(number):
    return round(number * 4) / 4


def round_up_to_even(number):
    return np.ceil(number / 2.) * 2


def zero_pad(y, Npad):
    """
    pads the beguining and end of y along the 1st axis with zeros
    """

    pads = np.zeros((Npad / 2,))
    yn = np.concatenate((pads, y, pads), axis=0)
    return yn


def fake_pad(y, Npad):
    """
    pads the beguining and end of y along the 1st axis with zeros
    """

    pads = np.ones((Npad / 2,)) * 9999
    yn = np.concatenate((pads, y, pads), axis=0)
    return yn


dameth = "linear"
dlm = 2.
di = 0
inputfile = ("proc_segs/cryosat_default_params/segments_dbs_os38_cryosat.npz")
lut = np.load(inputfile)["seg_dbase"]

short_lut = lut[lut['g_dist'] >= 550.]
short_lut = short_lut[np.logical_and(short_lut['lat_min'] <= -5.,
                      short_lut['lat_max'] >= -25.)]
shortest_lut = short_lut[np.logical_and(short_lut['lon_min'] <= -60.,
                         short_lut['lon_max'] >= -110.)]

dois = [45., 125.]
didx = np.empty((len(shortest_lut,), len(dois)), dtype=np.int)
# Idxs = np.empty((len(shortest_lut,)))
Idxs = []
dcover = np.empty((len(shortest_lut,)))
adep = np.empty((len(shortest_lut), len(dois)))
nngaps = np.empty((len(shortest_lut,), len(dois)))
# ngap_max = np.empty((len(shortest_lut,), len(dois)))
# ngap_tip = np.empty((len(shortest_lut,), len(dois)))
for n in range(0, len(shortest_lut)):
    tlon = shortest_lut['seg_data'][n].lon
    tlat = shortest_lut['seg_data'][n].lat
    ila = np.logical_and(tlat >= -25, tlat <= -5.)
    ilo = np.logical_and(tlon >= -110, tlon <= -74.)
    # idxs = np.logical_and(ila == True, ilo == True)
    idxs = np.logical_and(ila, ilo)
    # Idxs[n] = idxs
    Idxs.append(idxs)
    # if (idxs == False).all():
    if (idxs).all() is False:
        dcover[n] = 0
        for d in range(0, len(dois)):
            didx[n, d] = find_nearest(shortest_lut['dep'][n], dois[d])[1]
            nngaps[n, d] = 100.
            adep[n, d] = shortest_lut['seg_data'][n].depth[0, didx[n, d]]
    else:
        dcover[n] = shortest_lut['seg_data'][n].dl[idxs].sum()
        for d in range(0, len(dois)):
            didx[n, d] = find_nearest(shortest_lut['dep'][n], dois[d])[1]
            nmdpt = np.ma.count_masked(shortest_lut['seg_data'][n].u[idxs,
                                       didx[n, d]], axis=0)
            ntp = len(shortest_lut['seg_data'][n].u[idxs, didx[n, d]])
            nngaps[n, d] = 100 * nmdpt / ntp
            adep[n, d] = shortest_lut['seg_data'][n].depth[0, didx[n, d]]

# subset again the list based on the update info:
flut = shortest_lut[np.logical_and(dcover >= 950., nngaps[:, di] <= 12)]
Idxs = np.asanyarray(Idxs)
Idxs = Idxs[np.logical_and(dcover >= 950., nngaps[:, di] <= 12)]
didx = didx[np.logical_and(dcover >= 950., nngaps[:, di] <= 12)]

# Xis, Us, Vs, Es, ATs, XTs, = [], [], [], [], [], []
proc_segs_list = []
# need tuple (n, latmin, latmax, longmin, longmax)
for n, sg in enumerate(flut):
    lon, lat = sg['seg_data'].lon[Idxs[n]], sg['seg_data'].lat[Idxs[n]]
    u, v = sg['seg_data'].u[Idxs[n]], sg['seg_data'].v[Idxs[n]]
    uship, vship = sg['seg_data'].uship[Idxs[n]], sg['seg_data'].vship[Idxs[n]]
    dday, dep = sg['seg_data'].dday[Idxs[n]], sg['seg_data'].depth[Idxs[n]]
    errs = sg['seg_data'].errs[Idxs[n]]
    x = np.zeros(lat.shape)  # assumes no mask in lat/lon -> needs thought
    x[1:] = np.cumsum(sg['seg_data'].dl[Idxs[n]][1:])
    proc_seg_data = Bunch()

    # 1st deal with gaps:
    d1 = didx[n, di]
    ui = masked_interp(dday, u[:, d1][:, None])[:, 0]
    vi = masked_interp(dday, v[:, d1][:, None])[:, 0]
    erri = masked_interp(dday, errs[:, d1][:, None])[:, 0]

    lati = masked_interp(dday, lat[:, None])[:, 0]
    loni = masked_interp(dday, lon[:, None])[:, 0]

    # rotate vel (need to rethink, if not better done in shorter seg, because
    # interp), either use median uship/vship or low pass (ideal for long segs)
    if np.isscalar(uship.mask) is False:
        ushipi = masked_interp(dday, uship[:, None])[:, 0]
        vshipi = masked_interp(dday, vship[:, None])[:, 0]
    else:
        ushipi = uship
        vshipi = vship
    tp = round_up_to_even(24 * 3600 / sg['dt']) + 1  # 215 ~18hr
    uship_smo = smooth1d(np.squeeze(ushipi), int(tp))
    vship_smo = smooth1d(np.squeeze(vshipi), int(tp))
    # 2) get the smoothed angle:
    theta = np.angle(uship_smo + 1j*vship_smo)
    # 3) rotate:
    uv = ui + 1j*vi
    # atrack = ui[n]*np.cos(theta[:,None]) + vi[n]*np.sin(theta[:,None])
    # xtrack = -ui[n]*np.sin(theta[:,None]) + vi[n]*np.cos(theta[:,None])
    atrack = (uv * np.exp(-1j * theta)).real
    xtrack = (uv * np.exp(-1j * theta)).imag

    # 2nd put on a regular grid or block avg to coarser but filled low res:
    edist = x[-1]  # either this or the estimated g_dist?
    # dlm = np.ma.median()  # need to make it common to all
    xi = np.arange(0, np.floor(edist), dlm)
    fu = interp1d(x, ui, kind=dameth, axis=0)
    fv = interp1d(x, vi, kind=dameth, axis=0)
    fe = interp1d(x, erri, kind=dameth, axis=0)
    fa = interp1d(x, atrack, kind=dameth, axis=0)
    fx = interp1d(x, xtrack, kind=dameth, axis=0)
    AT, XT = fa(xi), fx(xi)
    U, V, E = fu(xi), fv(xi), fe(xi)
    fla = interp1d(x, lati, kind=dameth, axis=0)
    flo = interp1d(x, loni, kind=dameth, axis=0)
    lats, lons = fla(xi), flo(xi)

    # 500 km long uniform, with 50% overlap segments:
    n500 = int(500 / dlm)
    noverlap = int(.5 * n500)
    step = n500 - noverlap
    exs = len(xi) % n500
    topad = int(round_up_to_even(n500 - len(xi) % n500))
    if exs != 0 and exs <= topad:
        npts = len(xi[exs // 2:-exs // 2])
    elif exs != 0 and exs > topad:
        npts = topad + len(xi)
        U, V, E = zero_pad(U, topad), zero_pad(V, topad), zero_pad(E, topad)
        AT, XT = zero_pad(AT, topad), zero_pad(XT, topad)
        xi = zero_pad(xi, topad)  # can be used later to identify segs with pad
        lats = fake_pad(lats, topad)  # can be used later to identify segs padd
        lons = fake_pad(lons, topad)  # can be used later to identify segs padd
        if npts % 2 != 0:
            U, V, E = U[:-1], V[:-1], E[:-1]
            AT, XT, xi = AT[:-1], XT[:-1], xi[:-1]
            lats, lons = lats[:-1], lons[:-1]
    else:
        npts = len(xi)
    ind = np.arange(0, npts - n500 + 1, step)
    xis = np.empty((n500, len(ind)))
    lat_seg = np.empty((n500, len(ind)))
    lon_seg = np.empty((n500, len(ind)))
    u_seg = np.empty((n500, len(ind)))
    v_seg = np.empty((n500, len(ind)))
    e_seg = np.empty((n500, len(ind)))
    at_seg = np.empty((n500, len(ind)))
    xt_seg = np.empty((n500, len(ind)))
    for m in range(0, len(ind)):
        if exs != 0 and exs <= topad:
            xis[:, m] = xi[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            lat_seg[:, m] = lats[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            lon_seg[:, m] = lons[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            u_seg[:, m] = U[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            v_seg[:, m] = V[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            e_seg[:, m] = E[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            at_seg[:, m] = AT[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
            xt_seg[:, m] = XT[exs // 2:-exs // 2][ind[m]:ind[m] + n500]
        else:
            xis[:, m] = xi[ind[m]:ind[m]+n500]
            lat_seg[:, m] = lats[ind[m]:ind[m]+n500]
            lon_seg[:, m] = lons[ind[m]:ind[m]+n500]
            xis[:, m] = xi[ind[m]:ind[m]+n500]
            u_seg[:, m] = U[ind[m]:ind[m]+n500]
            v_seg[:, m] = V[ind[m]:ind[m]+n500]
            e_seg[:, m] = E[ind[m]:ind[m]+n500]
            at_seg[:, m] = AT[ind[m]:ind[m]+n500]
            xt_seg[:, m] = XT[ind[m]:ind[m]+n500]
    proc_seg_data.us = u_seg
    proc_seg_data.vs = v_seg
    proc_seg_data.es = e_seg
    proc_seg_data.ats = at_seg
    proc_seg_data.xts = xt_seg
    proc_seg_data.xis = xis
    proc_seg_data.lats = lat_seg
    proc_seg_data.lons = lon_seg
    proc_segs_list.append(proc_seg_data)
