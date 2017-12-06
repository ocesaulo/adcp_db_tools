#!/usr/bin/env python

'''

Script to extract segments of underway SADCP data from a number of sources,
store and categorize them in a lookup table.

List paths to all relevant CODAS databases to be "segmented" and iterate.

Load a CODAS ADCP data from final processing directory, either JAS or UHDAS
then perform the segmentation, then stores segmented data and metadata.

Segments are determined based on criterias of ship speed, duration of stops,
and location of points within a geodesic

A raw segment is all relevant CODAS ADCP data from a short transect determine

The meta-data contains: # and size of gaps; rotation angle and heading; length
of transect, average ship speed, rms error value, instrument type, ship name,
month of the segment, year of the segment, lat range, lon range, orientation
(meridional vs zonal), dt, dx/dy (dl), lat min, lat max, lon min, lon max

A processed segment is a: regularly gridded, gap-free (to the extent set), with
u, v, ur, vr (rotated coordinates), error, heading corrections, etc...
data structure (TBD); NOTE: orientatio will have to be revaluated again

Requires pycurrents.

TODO:

1. Timing and logging to file of some the prints and info
2. Argparsing
'''


# Relevant modules:
import numpy as np
# import matplotlib.pyplot as plt
import os
# import sys
import fnmatch
# import glob
# import argparse
# from matplotlib.dates import date2num, num2date, datestr2num
from pycurrents.codas import get_profiles
# from pycurrents.data.navcalc import lonlat_inside_km_radius
from pycurrents.data.navcalc import (great_circle_distance, diffxy_from_lonlat)
# from pycurrents.data.navcalc import great_circle_waypoints
from pycurrents.system import Bunch
from pycurrents.file import npzfile
from scipy.stats import mode as Mode


def read_meta(meta_file):
    dastrs = ('  sonar  ', '--sonar')
    with open(meta_file, 'r') as inF:
        for index, line in enumerate(inF):
            if dastrs[0] in line or dastrs[1] in line:
                # return line[-8:-1].strip()
                mystr = next(s for s in dastrs if s in line)
                # return line.split(mystr)[1].strip()
                sonar_line = line.split(mystr)[1].strip()
                if sonar_line.find(' ') == -1:
                    return sonar_line
                else:
                    return sonar_line[:sonar_line.find(' ')]


def read_bad_meta(meta_file):
    pt = ('narrowband', 'broadband', 'broad band', 'narrow band', 'Narrowband',
          'Broadband', 'Broad band', 'Narrow band', 'NarrowBand', 'BroadBand',
          'Broad Band', 'Narrow Band')
    with open(meta_file, 'r') as inF:
        for index, line in enumerate(inF):
            if ' HARDWARE MODEL ' in line:
                sonar_line = line[line.find(":")+1:-1].strip()
            if ' TRANSMIT FREQUENCY ' in line and any([p in line for p in pt]):
                cl = line[line.find(":")+1:-1]
                return (sonar_line + cl).strip()
            if ' COMMENTS ' in line and any([p in line for p in pt]):
                cl = line[line.find(":")+1:-1]
                return (sonar_line + cl).strip()
        return sonar_line


def translate_inst_str(astr):
    good_strs = ('os38nb', 'os38bb', 'os75nb', 'os75bb', 'os150nb', 'os150bb',
                 'wh300', 'nb150', 'bb150')
    if astr not in good_strs:
        if (
            'NB 150' in astr or 'VM-150' in astr or 'VM150' in astr or
            'NB-150' in astr or 'NB150' in astr or
            '150 kHz Narrowband' in astr
             ):
            return 'nb150'
        elif (
            'Broad Band 150' in astr or 'BB 150' in astr
             ):
            return 'bb150'
        elif (
            'Ocean Surveyor 75' in astr and 'narrowband' in astr
             ) or ('OS75' in astr and 'narrowband' in astr) or (
             '75KHz Ocean Surveyor' in astr and 'narrowband' in astr):
            return 'os75nb'
        elif (
            'Ocean Surveyor 75' in astr and 'broadband' in astr
             ) or ('OS75' in astr and 'broadband' in astr) or (
             '75KHz Ocean Surveyor' in astr and 'broadband' in astr):
            return 'os75bb'
        elif (
            'Ocean Surveyor 75' in astr and 'broadband' not in astr
             ) or ('OS75' in astr and 'broadband' not in astr) or (
             '75KHz Ocean Surveyor' in astr and 'broadband' not in astr or
            'Ocean Surveyor 75' in astr and 'narrowband' not in astr
             ) or ('OS75' in astr and 'narrowband' not in astr) or (
             '75KHz Ocean Surveyor' in astr and 'narrowband' not in astr
             ):
            return 'os75??'
        elif (
            'Ocean Surveyor 38' in astr and 'narrowband' in astr
             ) or ('OS38' in astr and 'narrowband' in astr) or (
             '38KHz Ocean Surveyor' in astr and 'narrowband' in astr) or (
             'Ocean Surveyer 38' in astr and 'narrowband' in astr) or (
             '38KHz Ocean Surveyer' in astr and 'narrowband' in astr):
            return 'os38nb'
        elif (
            'Ocean Surveyor 38' in astr and 'broadband' in astr
             ) or ('OS38' in astr and 'broadband' in astr) or (
             '38KHz Ocean Surveyor' in astr and 'broadband' in astr) or (
             'Ocean Surveyer 38' in astr and 'broadband' in astr) or (
             '38KHz Ocean Surveyer' in astr and 'broadband' in astr):
            return 'os38bb'
        else:
            return astr
    else:
        return astr


# Parameters (to be argparsed) to define and chop the segments (defaults):
mas = 3.  # minimum average ship speed during segment in m/s
tst = 1.  # tolarated stop time in hrs (longer will be split)
mtl = 50.  # minimum segment/transect length in km
# listfile = "dbs_os75_cryosat"
listfile = "dbs_os38_cryosat"
# listfile = "dbs_nb150_cryosat"

# outputing (some to be argparsed)
out_basedir = "proc_segs/"  # basic output directory; immutable
#  below should have a general descriptor name and idea of params used /
#  could be changed to outfilename:
user_out_dir = "cryosat_default_params/"  # also to be argparsed;
out_dir = out_basedir + user_out_dir  # output directory for seg. files and LUT
if listfile is None:
    outfilename = 'segments_' + user_out_dir[:-1] + '.npz'
else:
    outfilename = 'segments_' + listfile + '.npz'
if os.path.exists("./" + out_dir) is False:
    os.makedirs(out_dir)
print "Output directory and file for segments is ./" + out_dir + outfilename

if listfile is None:
    # finding the CODAS dbs and assembling the list of dbs to be segmented:
    parent_dir = os.getcwd()  # or to be argparsed

    # these reside in 2 dirs: jas_repo/sacid_dir and uhdas_repo/new_proc, a
    # crawl is necessary to find the paths to the dbs (must have *dir.blk file)
    dbslist = []

    for root, dirnames, filenames in os.walk(parent_dir):
        for filename in fnmatch.filter(filenames, '*dir.blk'):
            dbslist.append(os.path.join(root, filename[:-7]))
        # for filename in fnmatch.filter(filenames, '*dir.blk'):
        #     metalist.append()
else:
    dbslist = np.load(out_dir + listfile + '.npz')['dbslist'].tolist()
print "There are a total of ", len(dbslist), " dbs to process"

# iterate and segment:
lut = []  # this list will be appended with each segment (data/info)
# dbslist = dbslist[:3]

for d in dbslist:
    data = get_profiles(d, diagnostics=True)  # get all data

    # get meta-data (from where?? depends if JAS or UHDAS)
    if os.path.isfile(d + ".bft"):
        inst_info = read_meta(d + ".bft")
        if inst_info is None:
            inst_info = translate_inst_str(read_bad_meta(d + ".bft"))
    else:
        inst_info = read_meta(d[:-11] + "dbinfo.txt")
    # instlist.append(inst_info)

    svel = data.spd  # ship speed time series - need to ensure masks are same
    # lon = data.lon  # need to ensure masks are same NO!!! should be studied
    # lat = data.lat  # need to ensure masks are same how to handle diff masks!
    dtp = np.gradient(data.dday[svel > mas])  # time intervals when moving
    breaks = np.where(dtp > tst / 24.)[0]  # indices of start/end times of stop
    dts = round(np.ma.median(dtp) * 3600. * 24)
    # tmk = np.ma.masked_where(data.lo[svel > mas].mask, data.dday[svel > mas])

    # check if there are breaks:
    print "Are there breaks? ", len(breaks) != 0
    if len(breaks) != 0:
        # time and position of break points:
        # bdday = tmk[breaks]
        # blon = lon[svel > mas][breaks]
        # blat = lat[svel > mas][breaks]
        bdday = data.dday[svel > mas][breaks]
        blon = data.lon[svel > mas][breaks]
        blat = data.lat[svel > mas][breaks]

        # get geo distance between the breakpoints:
        g_dists = np.ma.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon[0], blat[0]),
                                1e-3 * great_circle_distance(blon[:-1],
                                blat[:-1], blon[1:], blat[1:]),
                                1e-3 * great_circle_distance(blon[-1],
                                blat[-1], data.lon[-1], data.lat[-1])))

        # get the indices of the original data where the break starts and ends:
        c = np.empty((g_dists.size + 1,), dtype=int)
        c[0], c[-1] = 0, -1
        for n in range(0, len(bdday)):
            c[n+1] = np.where(data.dday == bdday[n])[0][0]
    else:
        tmk = np.ma.masked_where(svel[svel > mas].mask, data.dday[svel > mas])
        bd = tmk.compressed()[0] - data.dday[0]
        be = data.dday[-1] - tmk.compressed()[-1]
        if bd > 0 and be > 0:
            # bslice = slice(np.where(data.dday == tmk.compressed()[0])[0][0],
            #                np.where(data.dday == tmk.compressed()[-1])[0][0])
            bslice = np.where(np.logical_or(data.dday == tmk.compressed()[0],
                              data.dday == tmk.compressed()[-1]))[0]
            blat = data.lat[bslice]
            blon = data.lon[bslice]
            bdday = data.dday[bslice]

            g_dists = np.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon[0], blat[0]),
                                1e-3*great_circle_distance(blon[:-1],
                                                           blat[:-1], blon[1:],
                                                           blat[1:]),
                                1e-3 * great_circle_distance(blon[-1],
                                                             blat[-1],
                                                             data.lon[-1],
                                                             data.lat[-1])))

            # get the indices of the original data
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[-1] = 0, -1
            for n in range(0, len(bdday)):
                c[n+1] = np.where(data.dday == bdday[n])[0][0]
        elif bd > 0 and be == 0:
            b1 = np.where(data.dday == tmk.compressed()[0])[0][0]
            blat = data.lat[b1]
            blon = data.lon[b1]
            bdday = data.dday[b1]

            g_dists = np.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon, blat),
                                1e-3 * great_circle_distance(blon, blat,
                                                             data.lon[-1],
                                                             data.lat[-1])))
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[1], c[-1] = 0, b1, -1
        elif bd == 0 and be > 0:
            b1 = np.where(data.dday == tmk.compressed()[-1])[0][0]
            blat = data.lat[b1]
            blon = data.lon[b1]
            bdday = data.dday[b1]

            g_dists = np.hstack((1e-3 * great_circle_distance(data.lon[0],
                                data.lat[0], blon, blat),
                                1e-3 * great_circle_distance(blon, blat,
                                                             data.lon[-1],
                                                             data.lat[-1])))
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[1], c[-1] = 0, b1, -1
        else:
            g_dists = np.array(1e-3 * great_circle_distance(data.lon[0],
                                                            data.lat[0],
                                                            data.lon[-1],
                                                            data.lat[-1]))
            c = np.empty((g_dists.size + 1,), dtype=int)
            c[0], c[-1] = 0, -1

    nsegs = len(g_dists)
    print "DB " + d + " has ", nsegs, " segments to evaluate"

    # loop through the final segments: save the data and info (into class)
    # checks of fast-tow and segment length to be left for later
    # ONLY CHECK: points must generally fall along the line/outliers discarded
    # outlier lies off the fake geodesic
    # CHECK: whether in between slowdowns the ship really stopped, e.g.
    # svel < .25 m/s - maybe be a bad criteria; currents can push it around
    # if the ship speed did not drop down to zero, it didn't stop, so ignore
    # if stopped, its station data, that must be ignored and the segment ends
    # these should correspond to segments that have g_dist -> 0
    # CHECK: for no valid data -> IMPORTANT
    counter = 0
    for n in range(0, nsegs):
        ndp = np.ma.count(svel[c[n]:c[n+1]])  # num of valid nav pts
        if ndp == 0:
            break
        g_dist = g_dists[n]
        a_spd = svel[c[n]:c[n+1]].mean()
        dcover = 1e-3 * np.sum(svel[c[n]:c[n+1]] * dts)  # should ~ g_dist idea
        sec_len_days = data["dday"][c[n+1]] - data["dday"][c[n]]
        lons, lats = data["lon"][c[n]:c[n+1]], data["lat"][c[n]:c[n+1]]
        dx, dy = diffxy_from_lonlat(lons, lats)
        dl = 1e-3 * np.sqrt(dx**2 + dy**2)

        # some tests must be made to know if its worth saving the seg data
        gndp = int(round(mtl / (dts * a_spd / 1e3)))
        # ndp = len(svel[c[n]:c[n+1]].compressed())  # num of valid nav pts
        dacond = (dcover >= mtl and g_dist >= mtl and
                  a_spd > mas and ndp >= gndp)

        if dacond:
            # figure out number of gaps and size of gaps (rms and max)
            nmdpt = np.ma.count_masked(data.u[c[n]:c[n+1]], axis=0)
            ngaps = 100. * nmdpt / len(data.u[c[n]:c[n+1]])
            # ngaps = np.zeros((data.dep.shape))
            gap_max = np.zeros((data.dep.shape))
            gap_tip = np.zeros((data.dep.shape))
            for k in range(0, len(data.dep)):
                # ngvdpt = len(data.u[c[n]:c[n+1]], k].compressed())
                # nvdpt = len(data.u[c[n]:c[n+1]], k])
                # ngaps[k] = (nvdpt - ngvdpt) / nvdpt
                gaps = np.ma.clump_masked(data.u[:, k])
                # gap_sizes = [len(np.arange(p.start, p.stop+1)) for p in gaps]
                gap_sizes = [np.ma.sum(dl[p]) for p in gaps]
                gap_max[k] = np.ma.max(gap_sizes)
                # gap_tip[k] = np.ma.median(gap_sizes)
                gap_tip[k] = Mode(gap_sizes)[0][0]

            seg_data = Bunch()
            seg_data.headings = data["heading"][c[n]:c[n+1]]
            seg_data.cogs = data["cog"][c[n]:c[n+1]]
            seg_data.lon, seg_data.lat, seg_data.dl = lons, lats, dl
            seg_data.svel = svel[c[n]:c[n+1]]
            seg_data.u = data["u"][c[n]:c[n+1]]
            seg_data.v = data["v"][c[n]:c[n+1]]
            seg_data.dday = data["dday"][c[n]:c[n+1]]
            seg_data.uship = data["uship"][c[n]:c[n+1]]
            seg_data.vship = data["vship"][c[n]:c[n+1]]
            seg_data.depth = data["depth"][c[n]:c[n+1]]
            seg_data.errs = data["e"][c[n]:c[n+1]]
            seg_data.ymdhms = data["ymdhms"][c[n]:c[n+1]]
            month = Mode(data.ymdhms[:, 1], axis=None)[0][0]
            lut.append((inst_info, data.yearbase, month, lats.min(),
                        lats.max(), lons.min(), lons.max(), g_dist, dcover,
                        sec_len_days, a_spd, data.dep, dts, np.ma.median(dl),
                        ngaps, gap_max, gap_tip, seg_data))
            counter = counter + 1
    # append LUT with a tuple that has descretized segment metadata, i.e.
    # things that can be searched for, then segment data as the last column
    print "final number of segments for this db is " + str(counter)
lut = np.array(lut, dtype=[("inst_id", '|S8'), ('year', 'int32'),
                           ('month', 'int32'), ('lat_min', 'float32'),
                           ('lat_max', 'float32'), ('lon_min', 'float32'),
                           ('lon_max', 'float32'), ('g_dist', 'float32'),
                           ('dcover', 'float32'), ('seg_days', 'float32'),
                           ('avg_spd', 'float32'), ('dep', 'O'),
                           ('dt', 'float16'), ('dlm', 'float16'),
                           ('ngaps', 'O'), ('gap_max', 'O'),
                           ('gap_tipical', 'O'),
                           ('seg_data', 'O')])

# save the look-up table and segment database (MAYBE PICKLE? if class/masked):
npzfile.savez(out_dir + outfilename, seg_dbase=lut)
print "Segment database saved to " + out_dir + outfilename
