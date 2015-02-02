# Plot results from testSynthetic.py
#
# Each plot is for 1 type of result.
#   (x, y) = (ntrain, result)
#   Plotlines: Different algs
#
# Plots saved as:
#   plotpath_stem + '.n[n].mu[w_mu].s[w_sigma2].a[a].[result_type]pdf'

import numpy, copy, math
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath('..'))
from addPaths import *
addPaths('..')

from syntheticTester import *
import index_builder

scriptname = sys.argv[0]
def usage():
    sys.stderr.write('usage: python %s [testpath]\n\n' % scriptname)
    raise Exception()

markersize = 6
legend_fontsize = 10

#=================================================================

if len(sys.argv) < 2:
    usage()
params = syntheticTesterParams(sys.argv[1])

NORMALIZE_W_G = False # If True, normalize w,g before comparing with truth.
VERBOSE = False
average_type = 'mean' # mean / median
normalize_std = True # If true, print stderr of mean (divide by sqrt(nruns)).

#=================================================================

result_types = ['L2_w_error', 'Linf_w_error', 'a_error', 'w_kendalltau']
btl_algs = set(['btl', 'btl-w'])
rbtl_algs = set(['rbtl-wa', 'rbtl-w-agrid', 'rbtl-a', 'rbtl-agrid', 'rbtl-w', 'rbtl'])
rbtl_indep_algs = set(['rbtl-indep-wg', 'rbtl-indep-g', 'rbtl-indep-w', 'rbtl-indep', \
                       'rbtl-indep-wg-wglink', 'rbtl-indep-g-wglink', 'rbtl-indep-w-wglink', 'rbtl-indep-wglink'])
if True:
    btl_algs = set(['btl'])
    rbtl_algs = set(['rbtl'])
    rbtl_indep_algs = set(['rbtl-indep', 'rbtl-indep-wglink'])
algs = []
algs += sorted(btl_algs)
algs += sorted(rbtl_algs)
algs += sorted(rbtl_indep_algs)
#algs = ['btl', 'btl-w', 'rbtl-wa', 'rbtl-a', 'rbtl-w', 'rbtl', \
#        'rbtl-indep-wg', 'rbtl-indep-g', 'rbtl-indep-w', 'rbtl-indep']
ntrain_vals = params.ntrain_vals
nruns = params.nruns

tmp_tester = syntheticTester(params) # for getting filepaths

result_type_index = index_builder.list2index(result_types)
alg_index = index_builder.list2index(algs)
result_type2long = {'L2_w_error' : 'L2(w^* - w) / n', \
                    'Linf_w_error' : 'Linf(w^* - w)', \
                    'a_error' : '|a^* - a|', \
                    'w_kendalltau' : 'KendallTau(w^*, w)'}
alg2label = {'btl'     : 'BTL',     \
             'btl-w'   : 'BTL-w',   \
             'rbtl-wa' : 'RBTL-wa', \
             'rbtl-w-agrid' : 'RBTL-w-agrid', \
             'rbtl-a'  : 'RBTL-a',  \
             'rbtl-agrid' : 'RBTL-agrid', \
             'rbtl-w'  : 'RBTL-w',  \
             'rbtl'    : 'RBTL',    \
             'rbtl-indep-wg' : 'RBTL-indep-wg', \
             'rbtl-indep-g'  : 'RBTL-indep-g',  \
             'rbtl-indep-w'  : 'RBTL-indep-w',  \
             'rbtl-indep'    : 'RBTL-indep', \
             'rbtl-indep-wg-wglink' : 'RBTL-indep-wglink-wg', \
             'rbtl-indep-g-wglink'  : 'RBTL-indep-wglink-g',  \
             'rbtl-indep-w-wglink'  : 'RBTL-indep-wglink-w',  \
             'rbtl-indep-wglink'    : 'RBTL-indep-wglink'}
alg2plottype = {'btl'           : '-k',     \
                'btl-w'         : ':k',   \
                'rbtl-wa'       : '-.r', \
                'rbtl-w-agrid'  : '-.*r', \
                'rbtl-a'        : '--r',  \
                'rbtl-agrid'    : '--*r',  \
                'rbtl-w'        : ':r',  \
                'rbtl'          : '-r',    \
                'rbtl-indep-wg' : '-.b', \
                'rbtl-indep-g'  : '--b',  \
                'rbtl-indep-w'  : ':b',  \
                'rbtl-indep'    : '-b', \
                'rbtl-indep-wg-wglink' : '-.g', \
                'rbtl-indep-g-wglink'  : '--g',  \
                'rbtl-indep-w-wglink'  : ':g',  \
                'rbtl-indep-wglink'    : '-g'}
for result_type in result_types:
    if result_type not in result_type2long:
        result_type2long[result_type] = result_type
        print 'WARNING: Using default long name for result_type: %s\n' % result_type

run2w = {} # run --> true w
missing_datafile_count = 0
for run in range(nruns):
    datapath = tmp_tester.getDataPath(params.datapath_stem, params.n, params.w_mu,\
                                      params.w_sigma2, params.theta[0], \
                                      params.theta[1], run)
    if not os.path.isfile(datapath):
        if VERBOSE:
            print 'WARNING: Missing data file: %s\n' % datapath
            missing_datafile_count += 1
        continue
    tester = jutils.read_pickle(datapath)
    run2w[run] = tester.truth.w
print '  Missing datafiles: %d' % missing_datafile_count

# Load results.
all_results = {} # alg --> result_type --> ntrain --> [results for each run]
missing_file_count = 0
bad_file_count = 0
good_file_count = 0
for alg in algs:
    print 'Loading data for alg = %s...' % alg
    alg_good_file_count = 0
    # Init all_results.
    all_results[alg] = {}
    for result_type in result_types:
        all_results[alg][result_type] = {}
        for ntrain in ntrain_vals:
            all_results[alg][result_type][ntrain] = []
    # Load results.
    for ntrain in ntrain_vals:
        for run in range(nruns):
            respath = \
              tmp_tester.getOutPathStem(params.outpath_stem, params.n, params.w_mu, \
                                        params.w_sigma2, params.theta[0], \
                                        params.theta[1], run, \
                                        alg, ntrain) \
              + '.pkl'
            if not os.path.isfile(respath):
                if VERBOSE:
                    print 'WARNING: Missing result file: %s\n' % respath
                missing_file_count += 1
                continue
            try:
                res = jutils.read_pickle(respath)
            except:
                print 'WARNING: Unreadable result file: %s\n' % respath
                bad_file_count += 1
                continue
            if NORMALIZE_W_G:
                learned_w = res['learned_model'].w
                w_normalizer = \
                  numpy.linalg.norm(learned_w,2) / numpy.linalg.norm(run2w[run],2)
                learned_w = learned_w / w_normalizer
            for result_type in result_types:
                if (result_type == 'a_error') and (alg not in rbtl_algs):
                    continue
                if NORMALIZE_W_G and (result_type == 'L2_w_error'):
                    val = numpy.linalg.norm(learned_w - run2w[run], 2) / params.n
                elif NORMALIZE_W_G and (result_type == 'Linf_w_error'):
                    val = numpy.max(numpy.abs(learned_w - run2w[run]))
                elif result_type in res:
                    val = res[result_type]
                else:
                    raise Exception('Bad result_type: %s\n' % result_type)
                all_results[alg][result_type][ntrain].append(val)
            good_file_count += 1
            alg_good_file_count +=1
    print '  For alg=%s, there were %d good files.' % (alg, alg_good_file_count)
print 'Read results.  File counts:'
print '  Good: %d' % good_file_count
print '  Bad: %d' % bad_file_count
print '  Missing: %d' % missing_file_count

# Plot results.
jutils.checkDirectory(params.plotpath_stem)
plotpath_dir = os.path.dirname(params.plotpath_stem)
plotpath_filestem = os.path.basename(params.plotpath_stem) + \
  ('.n%d.mu%g.s%g.a%g.b%g' % \
   (params.n, params.w_mu, params.w_sigma2, params.theta[0], params.theta[1]))
for result_type in result_types:
    plt.figure()
    for alg in algs:
        results = [] # for given (alg, result_type)
        stderrs = []
        ntrains_ = []
        for ntrain in ntrain_vals:
            result_vals = all_results[alg][result_type][ntrain]
            nruns_ = len(result_vals)
            if nruns_ <= 1:
                continue
            if average_type == 'mean':
                results.append(numpy.mean(result_vals))
                stderrs.append(numpy.std(result_vals))
            elif average_type == 'median':
                results.append(numpy.median(result_vals))
                stderrs.append(numpy.median(numpy.abs(result_vals - results[-1])))
            else:
                raise Exception('Bad average_type parameter: ' + average_type)
            if normalize_std:
                stderrs[-1] /= numpy.sqrt(nruns_)
            ntrains_.append(ntrain / float(params.n))
        if len(ntrains_) != 0:
            plt.errorbar(ntrains_, results, yerr=stderrs, fmt=alg2plottype[alg], label=alg2label[alg], markersize=markersize)
    plt.xlabel('# training examples / # students')
    plt.ylabel(result_type2long[result_type])
    title = os.path.basename(params.plotpath_stem) + '\n'
    if average_type == 'mean':
        title += 'Mean results.  Error bars: '
        if normalize_std:
            title += '1 stderr (std/sqrt(ntrials))'
            average_type_tag = 'mean_stderr'
        else:
            title += '1 std'
            average_type_tag = 'mean_std'
    elif average_type == 'median':
        title += 'Median results.  Error bars: '
        if normalize_std:
            title += '1 MAD / sqrt(ntrials)'
            average_type_tag = 'median_MADnorm'
        else:
            title += '1 MAD'
            average_type_tag = 'median_MAD'
    else:
        raise Exception('Bad average_type parameter: ' + average_type)
    if NORMALIZE_W_G and (result_type in ['L2_w_error', 'Linf_w_error']):
        title += '\n(w normalized to true L2norm)'
    plt.title(title)
    plt.gca().set_xscale('log')
    legend_loc = 1
    if result_type == 'w_kendalltau':
        legend_loc = 4
    plt.legend(loc=legend_loc, prop={'size':legend_fontsize})
    plotpath_tmp = plotpath_filestem + ('.%s.%s' % (average_type_tag, result_type))
    if NORMALIZE_W_G:
        plotpath_tmp += '.NORMW'
    plotpath = plotpath_dir + '/' + plotpath_tmp.replace('.', '_') + '.pdf'
    plt.savefig(plotpath)


