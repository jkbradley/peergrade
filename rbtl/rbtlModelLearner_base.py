"""
Base class for learning rbtlModel* variants.
"""

from rbtlModel_base import *


class rbtlModelLearnerParameters_base(object):
    def __init__(self):
        self.fix_w = False
        # w: prior
        self.w_prior = scoreDistribution('normal', [1.0,0.5])
        # w: constraints # TO DO: DO CV FOR CONSTRAINTS, IF NEEDED
        self.w_bounds = None  # For each student i, bounds [min, max] (Can be None)
        self.w_L2norm_constraint = None
    def copy(self, other):
        self.fix_w = other.fix_w
        self.w_prior = other.w_prior
        self.w_bounds = other.w_bounds
        self.w_L2norm_constraint = other.w_L2norm_constraint

class rbtlModelLearnerCVParameters_base(object):
    def __init__(self):
        # General CV params
        self.nfolds = 4
        self.cv_objective = 'll' # ll, acc
        self.cleanComparisons = False # Only use students who gave and received grades.
        # w: prior
        self.cv_choose_w_prior = True
        self.cv_grid_w_sigma2 = []
    def copy(self, other):
        self.nfolds = other.nfolds
        self.cv_objective = other.cv_objective
        self.cleanComparisons = other.cleanComparisons
        self.cv_choose_w_prior = other.cv_choose_w_prior
        self.cv_grid_w_sigma2 = other.cv_grid_w_sigma2
        
class rbtlModelLearner_base(object):

    def __init__(self, model, optParams, params=rbtlModelLearnerParameters_base()):
        self.model = model
        self.optParams = optParams
        # w
        self.fix_w = params.fix_w
        self.w_prior = params.w_prior
        self.w_bounds = params.w_bounds
        self.w_L2norm_constraint = params.w_L2norm_constraint

    # For 'normal':
    #  (.5/sigma2) * L2norm(w - mu)^2
    # For 'gamma':
    #  sum_i (alpha-1) log(w[i]) - w[i] / beta
    # @return  Regularization penalty for w.
    def regPenalty_w(self):
        if self.w_prior is not None:
            return self.w_prior.regPenalty(self.model.w)
        else:
            return 0

    def regGradient_w(self):
        if self.w_prior is not None:
            return self.w_prior.regGradient(self.model.w)
        else:
            return numpy.zeros(self.model.n)

    # Shared initialization for chooseRegularization
    def initChooseRegularization(self, CV_params):
        if CV_params.nfolds <= 1:
            raise Exception('chooseRegularization called with nfolds = %d' \
                            % CV_params.nfolds)
        if CV_params.cv_choose_w_prior:
            if self.w_prior is None:
                raise Exception('ERROR: chooseRegularization was told to choose w_prior, but was given no initial w_prior.')
            if len(CV_params.cv_grid_w_sigma2) == 0:
                raise Exception('ERROR: chooseRegularization was told to choose w_prior, but was given an empty cv_grid_w_sigma2.')
        else:
            if self.w_prior is None:
                CV_params.cv_grid_w_sigma2 = [0]
            else:
                CV_params.cv_grid_w_sigma2 = [self.w_prior.theta[1]]

