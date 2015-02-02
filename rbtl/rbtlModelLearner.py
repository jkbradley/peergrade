"""
Class for learning rbtlModel
"""

import sys
sys.path.append('../utils')
import jutils

from rbtlModel import *
from rbtlModelLearner_base import *


class rbtlModelLearnerParameters(rbtlModelLearnerParameters_base):
    def __init__(self, base_params=rbtlModelLearnerParameters_base()):
        super(rbtlModelLearnerParameters, self).copy(base_params)
        # alpha
        self.learn_alpha = 'grid' # fixed, grid, opt
        # alpha: choosing from grid
        self.alpha_grid = []
        # alpha: prior (for learn_alpha!='fixed')
        self.lambda_a = 0.01
    def __str__(self):
        s = '%s:\n' % (self.__class__)
        s += jutils.printDictAsString(self.__dict__, '  ')
        return s

class rbtlModelLearnerCVParameters(rbtlModelLearnerCVParameters_base):
    def __init__(self, base_params=rbtlModelLearnerCVParameters_base()):
        super(rbtlModelLearnerCVParameters, self).copy(base_params)
        # alpha: prior
        self.cv_choose_lambda_a = False # If learn_alpha != 'fixed'
        self.cv_grid_lambda_a = []
    def __str__(self):
        s = '%s:\n' % (self.__class__)
        s += jutils.printDictAsString(self.__dict__, '  ')
        return s


class rbtlModelLearner(rbtlModelLearner_base):

    # model: rbtlModel
    def __init__(self, model, optParams, params=rbtlModelLearnerParameters()):
        super(rbtlModelLearner, self).__init__(model, optParams, params)
        # alpha
        self.learn_alpha = params.learn_alpha
        self.alpha_grid = params.alpha_grid
        self.lambda_a = params.lambda_a

    # @return  Data log likelihood, minus regularization
    def objective(self, z_list):
        obj = self.model.logLikelihood(z_list)
        if not self.fix_w:
            obj -= self.regPenalty_w()
        if self.learn_alpha != 'fixed':
            obj -= self.regPenalty_a()
        return obj

    # Learn (theta, w).
    # @todo Try continuation to deal with non-convexity.
    def learn(self, z_list, record=None, log_fid=None, VERBOSE=False):
        self.model.theta = self.model.theta.astype(float)
        self.lambda_a = float(self.lambda_a)
        self.model.w = numpy.ones(self.model.n)
        # Set up record.
        #if record is not None:
        #    record.record(self, z_list)
        #    record.time_w.append(0)
        #    if not optParams.fix_theta:
        #        record.time_theta.append(0)
        # Optimize.
        nonNeg_w = False
        if (not self.fix_w) and (self.w_prior is not None):
            if self.w_prior.dist == 'normal':
                nonNeg_w = False
            elif self.w_prior.dist == 'gamma':
                nonNeg_w = True
            else:
                raise Exception('Bad prior on w: %s' % self.w_prior.dist)
        if VERBOSE and (log_fid is not None):
            print 'rbtlModelLearner.learn(): begin'
        now = time.time()
        self.optimize_CG(z_list, nonNeg_w, log_fid, VERBOSE)
        elapsed = time.time() - now
        if VERBOSE and (log_fid is not None):
            print 'rbtlModelLearner.learn(): complete after %d sec.' % elapsed
        # Record stuff.
        #if record is not None:
        #    record.time_w.append(elapsed)
        #    record.record(self, z_list)

    #-------------------------------------------------------
    # Internal methods
    #-------------------------------------------------------

    # @return  Regularization penalty for a.
    def regPenalty_a(self):
        return self.lambda_a * 0.5 * numpy.square(self.model.theta[0])

    def regGradient_a(self):
        return self.lambda_a * self.model.theta[0]

    # @return  Data log likelihood, minus regularization
    def objective_new_w_alpha(self, z_list, new_w, new_alpha):
        old_w = self.model.w
        old_alpha = self.model.theta[0]
        self.model.w = new_w
        self.model.theta[0] = new_alpha
        obj = self.model.logLikelihood(z_list) \
          - self.regPenalty_w() - self.regPenalty_a()
        self.model.w = old_w
        self.model.theta[0] = old_alpha
        return obj
    
    # Run CV to choose regularization.
    # This model instance is used as the base model, with pre-set parameters.
    # This instance is not modified.
    #
    # Arguments:
    #   z_list     Training data.
    #   CV_params  rbtlModelLearnerCVParameters
    #   log_fid    Log for printing CV info.
    # Returns:
    #   (best w_sigma2, best lambda_a)
    # (This also sets regularization in the stored model.)
    def chooseRegularization(self, z_list, CV_params, \
                             log_fid=sys.stderr, VERBOSE=False):
        # Initialize stuff.
        self.initChooseRegularization(CV_params)
        if self.learn_alpha != 'fixed':
            if CV_params.cv_choose_lambda_a:
                if len(CV_params.cv_grid_lambda_a) == 0:
                    raise Exception('Missing cv_grid_lambda_a for CV')
            else:
                CV_params.cv_grid_lambda_a = [self.lambda_a]
        else:
            CV_params.cv_grid_lambda_a = [0]
        
        if len(CV_params.cv_grid_w_sigma2) == 1 and len(CV_params.cv_grid_lambda_a) == 1:
            if VERBOSE:
                log_fid.write('WARNING: chooseRegularization called with nothing to choose from.\n')
        
        # Run CV.
        student_list = range(self.model.n)
        folds_z_lists = data_utils.splitFolds(z_list, len(student_list), CV_params.nfolds)
        test_objs = numpy.zeros((len(CV_params.cv_grid_w_sigma2), \
                               len(CV_params.cv_grid_lambda_a), \
                               CV_params.nfolds))
        orig_model = copy.deepcopy(self.model)
        total_runs = CV_params.nfolds * len(CV_params.cv_grid_w_sigma2) \
          * len(CV_params.cv_grid_lambda_a)
        num_runs = 0
        start_time = time.time()
        for fold in range(CV_params.nfolds):
            (fold_students, fold_train_z_list, fold_test_z_list) = \
              data_utils.getFoldi(student_list, folds_z_lists, fold, \
                                  cleanComparisons=CV_params.cleanComparisons)
            n = len(fold_students)
            for ws_i in range(len(CV_params.cv_grid_w_sigma2)):
                for la_i in range(len(CV_params.cv_grid_lambda_a)):
                    if (not self.fix_w) and (self.w_prior is not None):
                        # Use self's mean mu for w_prior.
                        self.w_prior.theta[1] = CV_params.cv_grid_w_sigma2[ws_i]
                    if self.learn_alpha != 'fixed':
                        self.lambda_a = CV_params.cv_grid_lambda_a[la_i]
                        self.model.setTheta([0,1])
                    self.learn(fold_train_z_list)#, log_fid=sys.stderr, VERBOSE=True)
                    if CV_params.cv_objective == 'll':
                        test_objs[ws_i,la_i,fold] = \
                          self.model.logLikelihood(fold_test_z_list) / float(len(fold_test_z_list))
                    elif CV_params.cv_objective == 'acc':
                        test_objs[ws_i,la_i,fold] = \
                          self.model.predictionAccuracy(fold_test_z_list) / float(len(fold_test_z_list))
                    num_runs += 1
                    if VERBOSE and (la_i == 0 and ws_i == 0):
                        elapsed = time.time() - start_time
                        expected_remaining = (elapsed / num_runs) * (total_runs - num_runs)
                        sys.stderr.write('rbtlModelLearner::chooseRegularization(): %d of %d runs took %d sec.  Expected time remaining = %d sec.\n' % (num_runs, total_runs, elapsed, expected_remaining))
        test_obj_means = numpy.mean(test_objs, 2)
        test_obj_stds = numpy.std(test_objs, 2) / numpy.sqrt(CV_params.nfolds)
        best_test_obj = numpy.max(test_obj_means)
        best_w_sigma2 = -1
        best_lambda_a = -1
        for ws_i in range(len(CV_params.cv_grid_w_sigma2)):
            for la_i in range(len(CV_params.cv_grid_lambda_a)):
                if test_obj_means[ws_i,la_i] == best_test_obj:
                    best_w_sigma2 = CV_params.cv_grid_w_sigma2[ws_i]
                    best_lambda_a = CV_params.cv_grid_lambda_a[la_i]
                    break
            if best_w_sigma2 != -1:
                break
        if log_fid != sys.stderr or VERBOSE:
            log_fid.write('\n')
            log_fid.write('CV results\n\n')
            log_fid.write('best_w_sigma2: %g\n' % best_w_sigma2)
            log_fid.write('best_lambda_a: %g\n' % best_lambda_a)
            log_fid.write('best_test_obj: %g\n' % best_test_obj)
            log_fid.write('\n')
            log_fid.write('Objective (%s) means (rows=w_sigma2, cols=lambda_a\n' % \
                          CV_params.cv_objective)
            jutils.print_table(log_fid, test_obj_means, CV_params.cv_grid_w_sigma2, CV_params.cv_grid_lambda_a)
            log_fid.write('\n')
            log_fid.write('Objective (%s) stderrs (rows=w_sigma2, cols=lambda_a\n' % \
                          CV_params.cv_objective)
            jutils.print_table(log_fid, test_obj_stds, CV_params.cv_grid_w_sigma2, CV_params.cv_grid_lambda_a)
            log_fid.write('\n')
        if (not self.fix_w) and (self.w_prior is not None):
            self.w_prior.theta[1] = best_w_sigma2
        if self.learn_alpha != 'fixed':
            self.lambda_a = best_lambda_a
        self.model = orig_model
        return (best_w_sigma2, best_lambda_a)


    # Optimize via CG or L-BFGS.
    def optimize_CG(self, z_list, nonNeg_w, log_fid=None, VERBOSE=False):
        n = self.model.n
        optParams = self.optParams
        init_params = \
          numpy.concatenate([numpy.ndarray.flatten(copy.deepcopy(self.model.w)), \
                             numpy.asarray([self.model.theta[0]])])
        eqcons = []
        fprime_eqcons = []
        if (not self.fix_w) and (self.w_L2norm_constraint is not None):
            cons_val = float(self.w_L2norm_constraint)
            eqcons = [lambda x,*args: numpy.array([numpy.linalg.norm(x[0:n],2)**2 - cons_val])]
            fprime_eqcons = (lambda x,*args: numpy.concatenate((numpy.array(2 * x[0:n]), [0])))
        w_bounds = [None,None]
        if not self.fix_w:
            if self.w_bounds is not None:
                if len(self.w_bounds) != 2:
                    raise Exception('Bad w_bounds parameter: %s' % str(self.w_bounds))
                w_bounds = self.w_bounds
            elif nonNeg_w:
                w_bounds = [0,None]
        if w_bounds == [None,None]:
            bounds = None
        else:
            bounds = [w_bounds] * n + [[None,None]]
        """
        learned_params,opt_val,info_dict = \
          optimize.fmin_l_bfgs_b(f_wrapper, init_params, \
                                 fprime=fprime_wrapper, \
                                 args=(model, z_list, optParams.fix_theta), \
                                 bounds=tmpbounds, factr=optParams.factr, \
                                 disp=disp)#, maxiter=optParams.max_iter)
        """
        if self.learn_alpha != 'grid':
            self.alpha_grid = [self.model.theta[0]]
        fix_alpha = (self.learn_alpha != 'opt')
        best_w_theta = copy.deepcopy(init_params)
        best_obj = self.objective(z_list)
        if VERBOSE and (log_fid is not None):
            print '  Init: L2norm(best_w_theta): %g' % numpy.linalg.norm(best_w_theta,2)
            print '        best_obj: %g' % best_obj
            print '        fix_alpha: ' + str(fix_alpha)
            if self.learn_alpha == 'grid':
                print '  Choosing alpha from grid:'
                print '  alpha\tobj\titerations'
            else:
                print '  init_alpha\tobj\titerations'
            full_output = True
        else:
            full_output = False
        for alpha in self.alpha_grid:
            self.model.theta[0] = alpha
            init_params[n] = alpha
            if bounds is None:
                # unconstrained
                learned_w_theta = \
                    optimize.fmin_cg(f_wrapper, init_params, \
                                     fprime=fprime_wrapper, \
                                     args=(self, z_list, self.fix_w, fix_alpha), \
                                     maxiter=optParams.max_iter, \
                                     disp=optParams.verbose)
            else:
                # constrained
                learned_w_theta = \
                    optimize.fmin_slsqp(f_wrapper, init_params, \
                                        eqcons=eqcons, bounds=bounds, \
                                        fprime=fprime_wrapper, \
                                        fprime_eqcons=fprime_eqcons, \
                                        args=(self, z_list, self.fix_w, fix_alpha), \
                                        iter=optParams.max_iter, acc=optParams.conv_tol, \
                                        disp=optParams.verbose, full_output=full_output)
                if full_output:
                    (learned_w_theta, final_obj, num_iters, opt_exit_mode, opt_exit_str) = \
                        learned_w_theta
                    print '  %g\t%g\t%d' % (alpha, final_obj, num_iters)
            obj = self.objective_new_w_alpha(z_list, learned_w_theta[0:n], learned_w_theta[n])
            if obj > best_obj:
                best_w_theta = copy.deepcopy(learned_w_theta)
                best_obj = obj
        self.model.w = best_w_theta[0:n]
        self.model.theta[0] = best_w_theta[n]
        tmp_obj = self.objective(z_list)
        if tmp_obj != best_obj:
            print 'WARNING: tmp_obj = %g, best_obj = %g\n' % (tmp_obj, best_obj)

# END OF class rbtlModelLearner



"""
Record of learning for gBTL model.
 - objective: ll, reg_w, reg_theta, total
 - iteration counts for w, theta
 - runtime
 - L2norm(parameter - truth) for w, theta
 - true model (for reference, when available)
 - truth['VALUE'] = value (when true model is available)
    where VALUE = train_LL, reg_w, reg_theta, n_malicious_graders, n_correct_exs
"""
class gBTLlearningRecord(object):

    def __init__(self, test_z_list = None):
        # Settings
        self.true_model = None  # true model, if available
        self.test_z_list = test_z_list  # test data, if available

        # Records for entire run (not over iterations):
        #--------------------------------------------------------------------------
        self.total_runtime = 0
        self.n_exs = 0

        # Records for each outer iteration (including before any iterations):
        # VALUE[outer iteration] = ...
        #--------------------------------------------------------------------------
        # Status:
        self.time_w = [] # runtime for optimizing w
        self.time_theta = []
        #   self.iterations_w = [] # inner iterations for optimizing w
        #   self.iterations_theta = []

        # Regularization penalties
        self.reg_w = [] # reg(w)
        self.reg_theta = [] # reg(theta)

        # Training data
        self.train_objective = [] # ll - reg
        self.train_LL = [] # ll

        # Test data
        self.test_LL = [] # ll

        # Error w.r.t. true_model
        self.truth = {} # See set_truth() for details.
        self.paramL2error_w = [] # L2norm(parameter - truth)
        self.paramL2error_theta = []

    def set_truth(self, true_model, train_z_list):
        self.true_model = true_model
        self.truth['reg_w'] = true_model.regPenalty_w()
        self.truth['reg_theta'] = true_model.regPenalty_a()
        self.truth['n_malicious_graders'] = numpy.sum(true_model.getAbilities() < 0)
        self.truth['train_LL'] = true_model.logLikelihood(train_z_list)
        self.truth['train_n_correct_exs'] = true_model.countCorrectExs(train_z_list)
        if self.test_z_list is not None:
            self.truth['test_LL'] = true_model.logLikelihood(self.test_z_list)
            self.truth['test_n_correct_exs'] = \
              true_model.countCorrectExs(self.test_z_list)

    # @param model   Learned model.
    # @todo Normalize stuff by n or n_exs (and adjust for that elsewhere in my code).
    def record(self, model, train_z_list):
        rs = model.regPenalty_w()
        rtheta = model.regPenalty_a()
        self.reg_w.append(rs)
        self.reg_theta.append(rtheta)

        train_ll = model.logLikelihood(train_z_list)
        self.train_LL.append(train_ll)
        self.train_objective.append(train_ll - rs - rtheta)

        if self.test_z_list is not None:
            test_ll = model.logLikelihood(self.test_z_list)
            self.test_LL.append(test_ll)

        if self.true_model is not None:
            self.paramL2error_w.append \
              (numpy.linalg.norm(model.w - self.true_model.w, 2))
            self.paramL2error_theta.append \
              (numpy.linalg.norm(model.theta - self.true_model.theta, 2))
# END OF class gBTLlearningRecord



# f for scipy.optimize (w,theta): neg log likelihood + regularization
# w_theta = [w, theta[0]]
def f_wrapper(w_theta, *args):
    learner, z_list, fix_w, fix_alpha = args
    model = learner.model
    if not fix_w:
        orig_w = model.w
        model.w = w_theta[0:model.n]
    if not fix_alpha:
        orig_alpha = model.theta[0]
        model.theta[0] = w_theta[model.n]
    tmpval = learner.objective(z_list)
    val = - tmpval
    #val = - learner.objective(z_list)
    if not fix_w:
        model.w = orig_w
    if not fix_alpha:
        model.theta[0] = orig_alpha
    return val

# gradient of f for scipy.optimize (w,theta)
# w_theta = [w, theta[0]]
def fprime_wrapper(w_theta, *args):
    learner, z_list, fix_w, fix_alpha = args
    model = learner.model
    if not fix_w:
        orig_w = model.w
        model.w = w_theta[0:model.n]
    if not fix_alpha:
        orig_alpha = model.theta[0]
        model.theta[0] = w_theta[model.n]
    g = model.getAbilities()
    (grad_theta, grad_w) = model.gradients(g, z_list)
    if fix_w:
        grad_w = numpy.zeros(model.n)
    else:
        grad_w -= learner.regGradient_w()
        model.w = orig_w
    if fix_alpha:
        grad_alpha = 0
    else:
        grad_alpha = grad_theta[0] - learner.regGradient_a()
        model.theta[0] = orig_alpha
    return - numpy.concatenate([grad_w, numpy.asarray([grad_alpha])])
