# Helper for running CV:
#
# Given dataset,
# * Run CV to choose regularization (inner_CV_nfolds).
# ** Estimate parameters from comparisons.
# ** Test on held-out comparisons.
# * Train on all data.
# * Test on held-out data, if any.

import sys, os, copy, time, datetime, pickle
import numpy, numpy.matlib

sys.path.append('..')
import addPaths
addPaths.addPaths('..')

import jutils, data_utils, loadParams
import optimizationParams
import rbtlModel_base, rbtlModel, rbtlModelIndep
import rbtlModelLearner_base, rbtlModelLearner, rbtlModelIndepLearner

#=================================================================

def printAndStore(dict, fid, key, val, prefix=''):
    dict[key] = val
    fid.write(prefix + '%s : %s\n' % (str(key), str(val)))

#=================================================================

# Parameters for runCVTest
class CVTestParameters(object):
    def __init__(self, inner_CV_nfolds, CV_objective='ll', \
                 conv_tol=None, max_iter=None):
        learner_base_params = \
            rbtlModelLearner_base.rbtlModelLearnerParameters_base()
        learner_base_params.w_prior.theta[0] = 1.0 # mu
        learner_base_CVparams = \
            rbtlModelLearner_base.rbtlModelLearnerCVParameters_base()
        learner_base_CVparams.nfolds = inner_CV_nfolds
        learner_base_CVparams.cv_objective = CV_objective
        learner_base_CVparams.cv_choose_w_prior = True
        learner_base_CVparams.cv_grid_w_sigma2 = [.001, .01, .1, 1, 10, 100]
        #   (for RBTL)
        self.init_theta = [0,1]
        self.rbtlLearner_params = \
            rbtlModelLearner.rbtlModelLearnerParameters(learner_base_params)
        self.rbtlLearner_params.learn_alpha = 'opt'
        self.rbtlLearner_CVparams = \
            rbtlModelLearner.rbtlModelLearnerCVParameters(learner_base_CVparams)
        self.rbtlLearner_CVparams.cv_choose_lambda_a = True
        self.rbtlLearner_CVparams.cv_grid_lambda_a = [.001, .01, .1, 1, 10, 100]
        #   (for RBTL-indep)
        self.init_g = 1
        self.rbtlIndepLearner_params = \
            rbtlModelIndepLearner.rbtlModelIndepLearnerParameters(learner_base_params)
        self.rbtlIndepLearner_params.g_prior.theta[0] = 1.0 # mu
        self.rbtlIndepLearner_CVparams = \
            rbtlModelIndepLearner.rbtlModelIndepLearnerCVParameters(learner_base_CVparams)
        self.rbtlIndepLearner_CVparams.cv_choose_g_sigma2 = True
        self.rbtlIndepLearner_CVparams.cv_grid_g_sigma2 = [.001, .01, .1, 1, 10, 100]
        # Optimization settings
        self.optParams = optimizationParams.optimizationParams()
        if conv_tol is not None:
            optParams.conv_tol = conv_tol
        if max_iter is not None:
            optParams.max_iter = max_iter

#=================================================================

"""
Parameters:
  n               Number of students
  train_z_list    Training data: [(i,j,l)] = [(i : j > l)]
  test_z_list     Testing data, if any.
  learned_model_type  rbtl, rbtl-indep, rbtl-indep-wglink
  outpath_stem        Records log in outpath_stem + '.log',
                      and results in outpath_stem + '.pkl'
  params              Dictionary of parameters saved to log and pkl files.
                       Note: 'scriptname' is treated specially
                        (printed first in log if exists).
  CVTest_params       CVTestParameters
"""
def runCVTest(n, train_z_list, test_z_list, learned_model_type, \
              outpath_stem, params, CVTest_params):
    ntrain = len(train_z_list)
    if test_z_list is not None:
        ntest = len(test_z_list)
    else:
        ntest = 0
    # Outfiles
    logpath = outpath_stem + '.log'
    outpath = outpath_stem + '.pkl'
    jutils.checkDirectory(logpath)
    if os.path.isfile(logpath):
        print 'SKIPPING existing result: %s' % logpath
        return
    else:
        os.system('touch %s' % logpath)
    log_fid = open(logpath, 'w')
    if 'scriptname' in params:
        log_fid.write('%s\n' % params['scriptname'])
    log_fid.write('\t %s\n' % str(datetime.datetime.now()))
    # Outfiles: Save parameters
    result = {}
    result['params'] = params
    log_fid.write('\nParameters:\n')
    jutils.printDict(params, log_fid, '  ', ignore=['scriptname'])
    log_fid.write('\n')
    # Outfiles: Save test info
    log_fid.write('Test info:\n')
    # SAVE TO params:
    printAndStore(result, log_fid, 'num_students', n, '  ')
    printAndStore(result, log_fid, 'ntrain', ntrain, '  ')
    printAndStore(result, log_fid, 'ntest', ntest, '  ')
    printAndStore(result, log_fid, 'learned_model_type', learned_model_type, '  ')
    log_fid.write('\n')
    # Set up model, learner.
    default_lambda_a_grid = CVTest_params.rbtlLearner_CVparams.cv_grid_lambda_a
    if learned_model_type == 'btl':
        learned = rbtlModel.rbtlModel(n)
        learned.setTheta([0,1])
        #
        learner_type = rbtlModelLearner.rbtlModelLearner
        learner_params = CVTest_params.rbtlLearner_params
        learner_params.learn_alpha = 'fixed'
        cv_params = CVTest_params.rbtlLearner_CVparams
        cv_params.cv_choose_lambda_a = False
    elif learned_model_type == 'rbtl':
        learned = rbtlModel.rbtlModel(n)
        learned.setTheta(CVTest_params.init_theta)
        #
        learner_type = rbtlModelLearner.rbtlModelLearner
        learner_params = CVTest_params.rbtlLearner_params
        learner_params.learn_alpha = 'opt'
        cv_params = CVTest_params.rbtlLearner_CVparams
        cv_params.cv_choose_lambda_a = True
        cv_params.cv_grid_lambda_a = default_lambda_a_grid
    elif learned_model_type in ['rbtl-indep', 'rbtl-indep-wglink']:
        learned = rbtlModelIndep.rbtlModelIndep(n)
        learned.g = numpy.ones(n) * CVTest_params.init_g
        #
        learner_type = rbtlModelIndepLearner.rbtlModelIndepLearner
        learner_params = CVTest_params.rbtlIndepLearner_params
        if learned_model_type == 'rbtl-indep':
            CVTest_params.rbtlIndepLearner_CVparams.cv_choose_lambda_wg = False
            CVTest_params.rbtlIndepLearner_CVparams.cv_grid_lambda_wg = [0]
        elif learned_model_type == 'rbtl-indep-wglink':
            CVTest_params.rbtlIndepLearner_CVparams.cv_choose_lambda_wg = True
            CVTest_params.rbtlIndepLearner_CVparams.cv_grid_lambda_wg = \
              [.001, .01, .1, 1, 10, 100]
        cv_params = CVTest_params.rbtlIndepLearner_CVparams
    else:
        raise Exception()
    learned.w = numpy.ones(n)
    #
    learner = learner_type(learned, CVTest_params.optParams, learner_params)
    # Outfiles: Save learner, CV params.
    log_fid.write('Learner params:\n')
    printAndStore(result, log_fid, 'learner_params', learner_params, '  ')
    printAndStore(result, log_fid, 'init_theta', CVTest_params.init_theta, '  ')
    printAndStore(result, log_fid, 'init_g', CVTest_params.init_g, '  ')
    printAndStore(result, log_fid, 'cv_params', cv_params, '  ')
    log_fid.write('\n')
    # Choose regularization, and train the model.
    start_time = time.time()
    learner.chooseRegularization(train_z_list, cv_params, \
                                 log_fid=log_fid, VERBOSE=True)
    printAndStore(result, log_fid, 'CV_runtime', time.time() - start_time)
    start_time = time.time()
    learner.learn(train_z_list, log_fid=log_fid, VERBOSE=True)
    printAndStore(result, log_fid, 'learning_runtime', time.time() - start_time)
    learned = learner.model
    # Outfiles: Save CV and learned info.
    result['learned_model'] = learned
    log_fid.write('Results:\n')
    printAndStore(result, log_fid, 'CV_w_sigma2', learner.w_prior.theta[1], '  ')
    if learned_model_type == 'btl':
        pass
    elif learned_model_type == 'rbtl':
        printAndStore(result, log_fid, 'CV_lambda_a', learner.lambda_a, '  ')
        printAndStore(result, log_fid, 'learned_alpha', learned.theta[0], '  ')
    elif learned_model_type in ['rbtl-indep', 'rbtl-indep-wglink']:
        printAndStore(result, log_fid, 'CV_g_sigma2', learner.g_prior.theta[1], '  ')
        printAndStore(result, log_fid, 'CV_lambda_wg', learner.lambda_wg, '  ')
    else:
        raise Exception('Bad learned_model_type: %s' % learned_model_type)
    # Compute results on test data.
    if ntest != 0:
        avg_testLL = learned.logLikelihood(test_z_list) / float(ntest)
        avg_testAcc = learned.predictionAccuracy(test_z_list) / float(ntest)
        printAndStore(result, log_fid, 'avg_testLL', avg_testLL, '  ')
        printAndStore(result, log_fid, 'avg_testAcc', avg_testAcc, '  ')
    # Save results.
    log_fid.close()
    jutils.dump_pickle(result, outpath)
    # Restore settings.
    cv_params.cv_grid_lambda_a = default_lambda_a_grid

