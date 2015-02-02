"""
Class for learning rbtlModelIndep
"""

import sys
sys.path.append('../utils')
import jutils

from rbtlModelIndep import *
from rbtlModelLearner_base import *


class rbtlModelIndepLearnerParameters(rbtlModelLearnerParameters_base):
    def __init__(self, base_params=rbtlModelLearnerParameters_base()):
        super(rbtlModelIndepLearnerParameters, self).copy(base_params)
        # g
        self.fix_g = False
        # g: prior
        self.g_prior = scoreDistribution('normal', [1.0,0.5])
        # g: constraints
        self.g_bounds = None  # For each student i, bounds [min, max] (Can be None)
        # w-g link: (1/2) * lambda_wg * L2norm(wg_alpha * w + 1 - g)^2
        self.lambda_wg = 0
        self.wg_alpha = 0
        self.wg_alpha_bounds = None
        # TO DO: Should we regularize wg_alpha, or is lambda_wg sufficient?
    def __str__(self):
        s = '%s:\n' % (self.__class__)
        s += jutils.printDictAsString(self.__dict__, '  ')
        return s

class rbtlModelIndepLearnerCVParameters(rbtlModelLearnerCVParameters_base):
    def __init__(self, base_params=rbtlModelLearnerCVParameters_base()):
        super(rbtlModelIndepLearnerCVParameters, self).copy(base_params)
        # g: prior
        self.cv_choose_g_prior = True
        self.cv_grid_g_sigma2 = []
        # w-g link
        self.cv_choose_lambda_wg = True
        self.cv_grid_lambda_wg = []
    def __str__(self):
        s = '%s:\n' % (self.__class__)
        s += jutils.printDictAsString(self.__dict__, '  ')
        return s

        
class rbtlModelIndepLearner(rbtlModelLearner_base):

    # model: rbtlModel
    def __init__(self, model, optParams, params=rbtlModelIndepLearnerParameters()):
        super(rbtlModelIndepLearner, self).__init__(model, optParams, params)
        # g
        self.fix_g = params.fix_g
        self.g_prior = params.g_prior
        self.g_bounds = params.g_bounds
        # w-g link
        self.lambda_wg = params.lambda_wg
        self.wg_alpha = params.wg_alpha
        self.wg_alpha_bounds = params.wg_alpha_bounds

    # @return  Data log likelihood, minus regularization
    def objective(self, z_list):
        obj = self.model.logLikelihood(z_list)
        if not self.fix_w:
            obj -= self.regPenalty_w()
        if not self.fix_g:
            obj -= self.regPenalty_g()
        if self.lambda_wg != 0:
            obj -= self.regPenalty_wg_link()
        return obj

    # Learn (w, g).
    def learn(self, z_list, log_fid=None, VERBOSE=False):
        self.model.w = self.model.w.astype(float)
        # Set up record.
        #    record.record(self, z_list)
        #    record.time_w.append(0)
        # Optimize.
        nonNeg_w = False
        if (not self.fix_w) and (self.w_prior is not None):
            if self.w_prior.dist == 'normal':
                nonNeg_w = False
            elif self.w_prior.dist == 'gamma':
                nonNeg_w = True
            else:
                raise Exception('Bad prior on w: %s' % self.w_prior.dist)
        nonNeg_g = False # Force this for now.
        now = time.time()
        delta = self.optimize_wg_CG(z_list, nonNeg_w, nonNeg_g)
        elapsed = time.time() - now
        #        record.time_iter.append(elapsed)

    #-------------------------------------------------------
    # Internal methods
    #-------------------------------------------------------
    
    # @return  Regularization penalty for g.
    def regPenalty_g(self):
        return self.g_prior.regPenalty(self.model.g)

    # @return  Gradient of regularization penalty for g.
    def regGradient_g(self):
        return self.g_prior.regGradient(self.model.g)

    # @return  Regularization penalty for w-g link:
    #             (1/2) * lambda_wg * L2norm(wg_alpha * w + 1 - g)^2
    def regPenalty_wg_link(self):
        return 0.5 * self.lambda_wg * \
          numpy.square(numpy.linalg.norm(self.wg_alpha * self.model.w + 1 - self.model.g))

    # @return  Gradient of regularization penalty for w-g link: (grad_w, grad_g, grad_wg_alpha)
    def regGradient_wg_link(self):
        tmp_grad = self.lambda_wg * (self.wg_alpha * self.model.w + 1 - self.model.g)
        return numpy.concatenate([self.wg_alpha * tmp_grad, \
                                  - tmp_grad, \
                                  numpy.asarray([numpy.inner(self.model.w, tmp_grad)])])

    # Run CV to choose regularization.
    # This model instance is used as the base model, with pre-set parameters.
    # This instance is not modified.
    #
    # Arguments:
    #   z_list     Training data.
    #   CV_params  rbtlModelIndepLearnerCVParameters
    #   log_fid    Log for printing CV info.
    # Returns:
    #   (best w_sigma2, best g_sigma2, best lambda_wg)
    #   (This also sets regularization in this learner class.)
    def chooseRegularization(self, z_list, CV_params, \
                             log_fid=sys.stderr, VERBOSE=False):
        # Initialize stuff.
        self.initChooseRegularization(CV_params)
        if not self.fix_g:
            if CV_params.cv_choose_g_prior:
                if self.g_prior is None:
                    raise Exception('Missing g_prior for CV')
                if len(CV_params.cv_grid_w_sigma2) == 0:
                    raise Exception('Missing cv_grid_w_sigma2 for CV')
            else:
                if self.g_prior is None:
                    CV_params.cv_grid_g_sigma2 = [0]
                else:
                    CV_params.cv_grid_g_sigma2 = [self.g_prior.theta[1]]
        else:
            CV_params.cv_grid_g_sigma2 = [0]
        if CV_params.cv_choose_lambda_wg:
            if len(CV_params.cv_grid_lambda_wg) == 0:
                raise Exception('Missing cv_grid_lambda_wg for CV')
        else:
            CV_params.cv_grid_lambda_wg = [self.lambda_wg]

        if len(CV_params.cv_grid_w_sigma2) == 1 and len(CV_params.cv_grid_g_sigma2) == 1 \
          and len(CV_params.cv_grid_lambda_wg) == 1:
            if VERBOSE:
                log_fid.write('WARNING: chooseRegularization called with nothing to choose from.\n')
        
        # Run CV.
        student_list = range(self.model.n)
        folds_z_lists = data_utils.splitFolds(z_list, len(student_list), CV_params.nfolds)
        test_objs = numpy.zeros((len(CV_params.cv_grid_w_sigma2), \
                               len(CV_params.cv_grid_g_sigma2), \
                               len(CV_params.cv_grid_lambda_wg), \
                               CV_params.nfolds))
        orig_model = copy.deepcopy(self.model)
        orig_wg_alpha = 0
        total_runs = CV_params.nfolds * len(CV_params.cv_grid_w_sigma2) \
          * len(CV_params.cv_grid_g_sigma2) * len(CV_params.cv_grid_lambda_wg)
        num_runs = 0
        start_time = time.time()
        for fold in range(CV_params.nfolds):
            (fold_students, fold_train_z_list, fold_test_z_list) = \
              data_utils.getFoldi(student_list, folds_z_lists, fold, \
                                  cleanComparisons=CV_params.cleanComparisons)
            n = len(fold_students)
            ntest = len(fold_test_z_list)
            for ws_i in range(len(CV_params.cv_grid_w_sigma2)):
                if (not self.fix_w) and (self.w_prior is not None):
                    # Use self's mean mu for w_prior.
                    self.w_prior.theta[1] = CV_params.cv_grid_w_sigma2[ws_i]
                for gs_i in range(len(CV_params.cv_grid_g_sigma2)):
                    if (not self.fix_g) and (self.g_prior is not None):
                        self.g_prior.theta[1] = CV_params.cv_grid_g_sigma2[gs_i]
                    for lwg_i in range(len(CV_params.cv_grid_lambda_wg)):
                        self.lambda_wg = CV_params.cv_grid_lambda_wg[lwg_i]
                        self.model = copy.deepcopy(orig_model)
                        self.wg_alpha = orig_wg_alpha
                        self.learn(fold_train_z_list)
                        if CV_params.cv_objective == 'll':
                            test_objs[ws_i,gs_i,lwg_i,fold] = \
                                self.model.logLikelihood(fold_test_z_list) / float(ntest)
                        elif CV_params.cv_objective == 'acc':
                            test_objs[ws_i,gs_i,lwg_i,fold] = \
                                self.model.predictionAccuracy(fold_test_z_list) / float(ntest)
                        num_runs += 1
                        if VERBOSE and (lwg_i == 0 and gs_i == 0 and ws_i == 0):
                            elapsed = time.time() - start_time
                            expected_remaining = \
                              (elapsed / num_runs) * (total_runs - num_runs)
                            sys.stderr.write('rbtlModelIndepLearner::chooseRegularization(): %d of %d runs took %d sec.  Expected time remaining = %d sec.\n' % (num_runs, total_runs, elapsed, expected_remaining))
        test_obj_means = numpy.mean(test_objs, 3)
        test_objstds = numpy.std(test_objs, 3) / numpy.sqrt(CV_params.nfolds)
        best_test_obj = numpy.max(test_obj_means)
        (best_w_sigma2,best_g_sigma2,best_lambda_wg) = (-1,-1,-1)
        for ws_i in range(len(CV_params.cv_grid_w_sigma2)):
            for gs_i in range(len(CV_params.cv_grid_g_sigma2)):
                for lwg_i in range(len(CV_params.cv_grid_lambda_wg)):
                    if test_obj_means[ws_i,gs_i,lwg_i] == best_test_obj:
                        best_w_sigma2 = CV_params.cv_grid_w_sigma2[ws_i]
                        best_g_sigma2 = CV_params.cv_grid_g_sigma2[gs_i]
                        best_lambda_wg = CV_params.cv_grid_lambda_wg[lwg_i]
                        break
                if best_w_sigma2 != -1:
                    break
            if best_w_sigma2 != -1:
                break
        if log_fid != sys.stderr or VERBOSE:
            log_fid.write('\n')
            log_fid.write('CV results\n\n')
            log_fid.write('best_w_sigma2: %g\n' % best_w_sigma2)
            log_fid.write('best_g_sigma2: %g\n' % best_g_sigma2)
            log_fid.write('best_lambda_wg: %g\n' % best_lambda_wg)
            log_fid.write('best_test_obj: %g\n' % best_test_obj)
            log_fid.write('\n\n')
            for lwg_i in range(len(CV_params.cv_grid_lambda_wg)):
                log_fid.write('Objective (%s) means (rows=w_sigma2, cols=g_sigma2), lambda_wg=%g\n' % \
                              (CV_params.cv_objective, CV_params.cv_grid_lambda_wg[lwg_i]))
                jutils.print_table(log_fid, test_obj_means[:,:,lwg_i], \
                                   CV_params.cv_grid_w_sigma2, CV_params.cv_grid_g_sigma2)
                log_fid.write('\n')
                log_fid.write('Objective (%s) stderrs (rows=w_sigma2, cols=g_sigma2), lambda_wg=%g\n' % \
                              (CV_params.cv_objective, CV_params.cv_grid_lambda_wg[lwg_i]))
                jutils.print_table(log_fid, test_objstds[:,:,lwg_i], \
                                   CV_params.cv_grid_w_sigma2, CV_params.cv_grid_g_sigma2)
                log_fid.write('\n')
            log_fid.write('\n')
        if best_w_sigma2 == -1 or best_g_sigma2 == -1:
            raise Exception('CROSSVAL FAILED! (best_w_sigma2, best_g_sigma2, best_lambda_wg) = (%g, %g, %g)' % (best_w_sigma2, best_g_sigma2, best_lambda_wg))
        if (not self.fix_w) and (self.w_prior is not None):
            self.w_prior.theta[1] = best_w_sigma2
        if (not self.fix_g) and (self.g_prior is not None):
            self.g_prior.theta[1] = best_g_sigma2
        self.lambda_wg = best_lambda_wg
        self.model = orig_model
        self.wg_alpha = orig_wg_alpha
        return (best_w_sigma2, best_g_sigma2, best_lambda_wg)

    
    # Optimize w and g.
    # @return  max_delta (change in w,g)
    def optimize_wg_CG(self, z_list, nonNeg_w, nonNeg_g):
        n = self.model.n
        optParams = self.optParams
        init_params = numpy.concatenate([copy.deepcopy(self.model.w), \
                                         copy.deepcopy(self.model.g), \
                                         numpy.asarray([self.wg_alpha])])
        eqcons = []
        fprime_eqcons = []
        if (not self.fix_w) and (self.w_L2norm_constraint is not None):
            cons_val = float(self.w_L2norm_constraint)
            eqcons = [lambda x,*args: numpy.array([numpy.linalg.norm(x[0:n],2)**2 - cons_val])]
            fprime_eqcons = \
              (lambda x,*args: numpy.concatenate((numpy.array(2 * x[0:n]), numpy.zeros(n))))
        w_bounds = [None,None]
        if not self.fix_w:
            if self.w_bounds is not None:
                if len(self.w_bounds) != 2:
                    raise Exception('Bad w_bounds parameter: %s' % str(self.w_bounds))
                w_bounds = self.w_bounds
            elif nonNeg_w:
                w_bounds = [0,None]
        g_bounds = [None,None]
        if not self.fix_g:
            if self.g_bounds is not None:
                if len(self.g_bounds) != 2:
                    raise Exception('Bad g_bounds parameter: %s' % str(self.g_bounds))
                g_bounds = self.g_bounds
            elif nonNeg_g:
                g_bounds = [0,None]
        wg_alpha_bounds = [None,None]
        if self.lambda_wg != 0:
            if self.wg_alpha_bounds is not None:
                if len(self.wg_alpha_bounds) != 2:
                    raise Exception('Bad wg_alpha_bounds parameter: %s' % \
                                    str(self.wg_alpha_bounds))
                wg_alpha_bounds = self.wg_alpha_bounds
        if w_bounds == [None,None] and g_bounds == [None,None] \
          and wg_alpha_bounds == [None,None]:
            bounds = None
        else:
            bounds = [w_bounds] * n + [g_bounds] * n + [wg_alpha_bounds]
        """
        if nonNeg_w or nonNeg_g or \
          (self.w_bounds is not None) or (self.g_bounds is not None):
            # constrained
            tmpbounds = []
            if self.w_bounds is not None:
                tmpbounds = [self.w_bounds] * n
            elif nonNeg_w:
                tmpbounds = [[0.01,100]] * n
            else:
                tmpbounds = [[None,None]] * n
            if self.g_bounds is not None:
                tmpbounds += [self.g_bounds] * n
            elif nonNeg_g:
                tmpbounds += [[0.01,100]] * n
            else:
                tmpbounds += [[None,None]] * n
            w_g = optimize.fmin_l_bfgs_b(f_wrapper, init_params, \
                                         fprime=fprime_wrapper, \
                                         args=[self.model, z_list], \
                                         bounds=tmpbounds, factr=optParams.factr)
                                         #, maxiter=optParams.max_iter)
            w_g = w_g[0] # ignore extra info returned by optimize
        else:
        """
        if bounds is None:
            # unconstrained
            learned_w_g_wgalpha = \
              optimize.fmin_cg(f_wrapper, init_params, \
                               fprime=fprime_wrapper, \
                               args=(self, z_list, self.fix_w, self.fix_g), \
                               maxiter=optParams.max_iter, \
                               disp=optParams.verbose)
        else:
            # constrained
            learned_w_g_wgalpha = \
                optimize.fmin_slsqp(f_wrapper, init_params, \
                                    eqcons=eqcons, bounds=bounds, \
                                    fprime=fprime_wrapper, \
                                    fprime_eqcons=fprime_eqcons, \
                                    args=(self, z_list, self.fix_w, self.fix_g), \
                                    iter=optParams.max_iter, acc=optParams.conv_tol, \
                                    disp=optParams.verbose)
        learned_w = learned_w_g_wgalpha[:n]
        learned_g = learned_w_g_wgalpha[n:2*n]
        learned_wg_alpha = learned_w_g_wgalpha[2*n]
        max_delta = max(numpy.max(numpy.abs(self.model.w - learned_w)), \
                        numpy.max(numpy.abs(self.model.g - learned_g)), \
                        numpy.abs(self.wg_alpha - learned_wg_alpha))
        self.model.w = learned_w
        self.model.g = learned_g
        self.wg_alpha = learned_wg_alpha
        return max_delta

# END OF class rbtlModelIndepLearner


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
        self.truth['n_malicious_graders'] = numpy.sum(true_model.computeAbilities() < 0)
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


# f for fmin_cg(w,g): neg log likelihood + regularization
def f_wrapper(w_g, *args):
    learner, z_list, fix_w, fix_g = args
    model = learner.model
    n = model.n
    #
    if not fix_w:
        orig_w = model.w
        model.w = w_g[0:n]
    if not fix_g:
        orig_g = model.g
        model.g = w_g[n:2*n]
    orig_wg_alpha = learner.wg_alpha
    learner.wg_alpha = w_g[2*n]
    #
    val = - learner.objective(z_list)
    #
    if not fix_w:
        model.w = orig_w
    if not fix_g:
        model.g = orig_g
    learner.wg_alpha = orig_wg_alpha
    return val

# gradient of f for fmin_cg(w,g)
def fprime_wrapper(w_g, *args):
    learner, z_list, fix_w, fix_g = args
    model = learner.model
    n = model.n
    #
    if not fix_w:
        orig_w = model.w
        model.w = w_g[0:n]
    if not fix_g:
        orig_g = model.g
        model.g = w_g[n:2*n]
    orig_wg_alpha = learner.wg_alpha
    learner.wg_alpha = w_g[2*n]
    #
    (grad_w, grad_g) = model.gradients(z_list)
    if fix_w:
        grad_w = numpy.zeros(n)
    else:
        grad_w -= learner.regGradient_w()
    if fix_g:
        grad_g = numpy.zeros(n)
    else:
        grad_g -= learner.regGradient_g()
    grad_full = numpy.concatenate([grad_w,grad_g,numpy.asarray([0.])])
    if learner.lambda_wg != 0:
        grad_full -= learner.regGradient_wg_link()
    #
    if not fix_w:
        model.w = orig_w
    if not fix_g:
        model.g = orig_g
    learner.wg_alpha = orig_wg_alpha
    return - grad_full

