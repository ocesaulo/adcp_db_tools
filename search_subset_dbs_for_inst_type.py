#!/usr/bin/env python

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
# from pycurrents.data.navcalc import great_circle_waypoints
# from pycurrents.system import Bunch
# from pycurrents.file import npzfile
# from scipy.stats import mode as Mode


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


# inst type (string bit) you want to get shortlisted:
inst_type = 'nb150'

# outputing (some to be argparsed)
out_basedir = "proc_segs/"  # basic output directory; immutable
#  below should have a general descriptor name and idea of params used /
#  could be changed to outfilename:
user_out_dir = "cryosat_default_params/"  # also to be argparsed;
out_dir = out_basedir + user_out_dir  # output directory for seg. files and LUT
outfilename = 'dbs_' + inst_type + '_' + 'cryosat' + '.npz'
if os.path.exists("./" + out_dir) is False:
    os.makedirs(out_dir)
print "Output directory and file for segments is ./" + out_dir + outfilename

# finding the CODAS databases and assembling the list of dbs to be segmented:
parent_dir = os.getcwd()  # or to be argparsed

# these reside in two directories: jas_repo/sacid_dir and uhdas_repo/new_proc,
# a crawl is necessary to find the paths to the dbs (must have *dir.blk file):
dbslist = []

for root, dirnames, filenames in os.walk(parent_dir):
    for filename in fnmatch.filter(filenames, '*dir.blk'):
        dbslist.append(os.path.join(root, filename[:-7]))
print "There are a total of ", len(dbslist), " dbs to process"

# iterate and segment:
instlist = []

for d in dbslist:
    data = get_profiles(d, diagnostics=True)  # get all data

    # get meta-data (from where?? depends if JAS or UHDAS)
    if os.path.isfile(d + ".bft"):
        inst_info = read_meta(d + ".bft")
        if inst_info is None:
            inst_info = translate_inst_str(read_bad_meta(d + ".bft"))
    else:
        inst_info = read_meta(d[:-11] + "dbinfo.txt")
    instlist.append(inst_info)

shortlist = [dbslist[i] for i, e in enumerate(instlist) if inst_type in e]
# os38list = [l for l in instlist if '38' in l]
np.savez(out_dir + outfilename, dbslist=shortlist)
