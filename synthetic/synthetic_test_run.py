# Compare several models (BTL, RBTL, etc.).
#
# Test:
#   Fix learning problem size.
#   For several runs,
#     Generate a model.
#     Generate training data.
#     For a range of ntrain,
#       Train models.  Compare results.
#
# Data files saved as:
#   datapath_stem + '.n[n].mu[w_mu].s[w_sigma2].a[a].run[run].pkl'
# Result files saved as:
#   outpath_stem + '.n[n].mu[w_mu].s[w_sigma2].a[a].run[run].[alg].ntrain[ntrain].pkl'

import numpy

import sys, os
sys.path.append(os.path.abspath('..'))
from addPaths import *
addPaths('..')

from syntheticTester import *

scriptname = sys.argv[0]
def usage():
    sys.stderr.write('usage: python %s [testpath] ([CLEAN])\n\n' % scriptname)
    raise Exception()

#=================================================================

if len(sys.argv) < 2:
    usage()
params = syntheticTesterParams(sys.argv[1])
CLEAN = False
if len(sys.argv) >= 3:
    if sys.argv[2] == 'CLEAN':
        CLEAN = True
    else:
        usage()

ntrain_vals = params.ntrain_vals
nruns = params.nruns
n = params.n
w_mu = params.w_mu
w_sigma2 = params.w_sigma2
theta = params.theta

#=================================================================
# FIXED PARAMETERS

run_btl = True
run_rbtlModel = True
run_rbtlModelIndep = True

#=================================================================

def runOne(outpath_stem, tester, ntrain, CLEAN):
    outpath = outpath_stem + '.pkl'
    logpath = outpath_stem + '.log'
    if os.path.isfile(logpath):
        if CLEAN:
            bad = False
            try:
                res = jutils.read_pickle(outpath)
            except:
                bad = True
            if bad:
                print 'DELETING bad result: %s\n' % logpath
                if os.path.isfile(outpath):
                    os.rename(outpath, outpath + '.DELETED')
                if os.path.isfile(logpath):
                    os.rename(logpath, logpath + '.DELETED')
        else:
            print 'SKIPPING existing result: %s\n' % logpath
        return
    jutils.checkDirectory(logpath)
    os.system('touch %s' % logpath)
    log_fid = open(logpath, 'w')
    log_fid.write(scriptname + '\n')
    log_fid.write('\t %s\n' % str(datetime.datetime.now()))
    log_fid.write('%s\n' % str(tester))
    log_fid.write('ntrain : %d\n' % ntrain)
    res = tester.learn(ntrain, log_fid)
    log_fid.close()
    jutils.dump_pickle(res, outpath)

#=================================================================

tester = syntheticTester(params)
tester.fixLambda_a = False
tester.VERBOSE = False

for run in range(nruns):
    print 'synthetic_test_run: Starting run %d\n' % run
    # Load or create model, data.
    datapath = \
      tester.getDataPath(params.datapath_stem, n, w_mu, w_sigma2, theta[0], theta[1], run)
    if os.path.isfile(datapath):
        print 'LOADING existing data: %s\n' % datapath
        tester = jutils.read_pickle(datapath)
    else:
        jutils.checkDirectory(datapath)
        os.system('touch %s' % datapath)
        tester.createTruth(theta)
        tester.genData(params.max_ntrain)
        jutils.dump_pickle(tester, datapath)
    for ntrain_i in range(len(ntrain_vals)):
        ntrain = ntrain_vals[ntrain_i]
        print 'synthetic_test_run: Starting run %d, ntrain %d\n' % (run, ntrain)

        # BTL Variants
        if run_btl:
            tester.p.learned_model_type = 'rbtl'
            tester.p.learned_fixedTheta = [0,1]
            tester.p.rbtlModelLearner_params.learn_alpha = 'fixed'
            #   BTL
            tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = True
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run, 'btl', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            if tester.w_prior is not None:
                #   BTL, sigma2 fixed at truth
                tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = False
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'btl-w', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)

        # RBTL Variants
        if run_rbtlModel:
            tester.p.learned_model_type = 'rbtl'
            tester.p.learned_fixedTheta = None
            #   RBTL, nothing fixed
            tester.p.rbtlModelLearner_params.learn_alpha = 'opt'
            tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = True
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            #   RBTL, alpha grid
            tester.p.rbtlModelLearner_params.learn_alpha = 'grid'
            tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = True
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl-agrid', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            #   RBTL, theta fixed
            tester.p.rbtlModelLearner_params.learn_alpha = 'fixed'
            tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = True
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl-a', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            if tester.w_prior is not None:
                #   RBTL, sigma2 fixed
                tester.p.rbtlModelLearner_params.learn_alpha = 'opt'
                tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = False
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-w', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)
                #   RBTL, sigma2 fixed, alpha grid
                tester.p.rbtlModelLearner_params.learn_alpha = 'grid'
                tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = False
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-w-agrid', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)
                #   RBTL, theta and sigma2 fixed at truth
                tester.p.rbtlModelLearner_params.learn_alpha = 'fixed'
                tester.p.rbtlModelLearner_CVparams.cv_choose_w_prior = False
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-wa', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)

        if run_rbtlModelIndep:
            tester.p.learned_model_type = 'rbtlIndep'
            tester.p.learned_fixedTheta = None
            #
            # RBTL-Indep Variants: without w-g link
            #
            tester.p.rbtlModelIndepLearner_params.lambda_wg = 0
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_lambda_wg = False
            #   RBTL-indep, nothing fixed
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = True
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = True
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl-indep', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            #   RBTL-indep, g_sigma2 fixed at truth
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = True
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = False
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl-indep-g', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            if tester.w_prior is not None:
                #   RBTL-indep, sigma2 fixed at truth
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = False
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = True
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-indep-w', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)
                #   RBTL-indep, sigma2 and g_sigma2 fixed at truth
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = False
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = False
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-indep-wg', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)
            #
            # RBTL-Indep Variants: with w-g link
            #
            tester.p.rbtlModelIndepLearner_params.lambda_wg = 0
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_lambda_wg = True
            #   RBTL-indep, nothing fixed
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = True
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = True
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl-indep-wglink', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            #   RBTL-indep, g_sigma2 fixed at truth
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = True
            tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = False
            outpath_stem = \
              tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                    theta[0], theta[1], run,'rbtl-indep-g-wglink', ntrain)
            runOne(outpath_stem, tester, ntrain, CLEAN)
            if tester.w_prior is not None:
                #   RBTL-indep, sigma2 fixed at truth
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = False
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = True
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-indep-w-wglink', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)
                #   RBTL-indep, sigma2 and g_sigma2 fixed at truth
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_w_prior = False
                tester.p.rbtlModelIndepLearner_CVparams.cv_choose_g_prior = False
                outpath_stem = \
                  tester.getOutPathStem(params.outpath_stem, n, w_mu, w_sigma2, \
                                        theta[0], theta[1], run,'rbtl-indep-wg-wglink', ntrain)
                runOne(outpath_stem, tester, ntrain, CLEAN)
