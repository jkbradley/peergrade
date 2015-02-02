# Various utility functions.

import pickle, random, os
import numpy as np

import index_builder

#import asciitable, copy

# Print m x n table, with row and column labels
# Arguments:
#   table        m x n table
#   row_labels   m labels (optional)
#   col_labels   n labels (optional)
def print_table(fid, table, row_labels = None, col_labels = None):
    (m,n) = np.shape(table)
    if col_labels is not None:
        if row_labels is not None:
            fid.write('\t')
        fid.write('\t'.join(map(str,col_labels)) + '\n')
    for i in range(m):
        if row_labels is not None:
            fid.write(str(row_labels[i]) + '\t')
        fid.write('\t'.join(map(str, table[i,])) + '\n')

# Read 1 item from a pickle file.
def read_pickle(filepath):
    try:
        fid = open(filepath, 'rb')
        dat = pickle.load(fid)
        fid.close()
    except:
        raise Exception('Could not read file: ' + filepath)
    return dat

# Save 1 item to a pickle file.
def dump_pickle(item, filepath):
    try:
        fid = open(filepath, 'wb')
        pickle.dump(item, fid, -1)
        fid.close()
    except:
        raise Exception('Could not write file: ' + filepath)

# Returns intersection of keys of 2 dictionaries.
# Values are taken from the first dictionary.
def dict_intersect(d1, d2):
    d = {}
    if len(d1) < len(d2):
        for i in d1:
            if i in d2:
                d[i] = d1[i]
    else:
        for i in d2:
            if i in d1:
                d[i] = d1[i]
    return d

# Returns union of keys of 2 dictionaries.
# Values are taken from the first dictionary when available.
def dict_union(d1, d2):
    d = d1
    for i in d2:
        if i not in d:
            d[i] = d2[i]
    return d

# Print dictionary as list of lines, sorted by keys:
# [key] : [value]
#
# Arguments:
#   d       Dictionary
#   fid     File ID to print to.  If fid is not given, print to STDOUT.
#   prefix  Prefix each line with this.
#   ignore  Do not print any entries with keys in this list.
# Returns: string
def printDict(d, fid=None, prefix='', ignore=[]):
    if fid is None:
        fid = sys.stdout
    fid.write(printDictAsString(d, prefix, ignore))

# Print dictionary to string as list of lines, sorted by keys:
# [key] : [value]
# Arguments:
#   d       Dictionary
#   prefix  Prefix each line with this.
#   ignore  Do not print any entries with keys in this list.
# Returns: string
def printDictAsString(d, prefix='', ignore=[]):
    ignore_set = set(ignore)
    s = ''
    for k in sorted(d.keys()):
        if k not in ignore_set:
            s += prefix + str(k) + '\t:\t' + str(d[k]) + '\n'
    return s

# Return a random non-overlapping triplet (i,j,l) in [0,n).
def randTriplet(n):
    # TO DO: FIND RANDPERM FUNCTION
    if n < 3:
        raise Exception('randTriplet called with n<3!')
    i = random.randint(0,n-1)
    j = random.randint(0,n-1)
    while j == i:
        j = random.randint(0,n-1)
    l = random.randint(0,n-1)
    while (l == i) or (l == j):
        l = random.randint(0,n-1)
    return (i,j,l)

# Check if the bottom-most directory in the path exists.
# If not, create it.
def checkDirectory(path, VERBOSE=True):
    d = os.path.dirname(path)
    if os.path.isdir(d):
        return
    os.mkdir(d)
    if VERBOSE:
        print 'Created directory: %s' % d

