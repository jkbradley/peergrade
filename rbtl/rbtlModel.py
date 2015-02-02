"""
Generalized Bradley-Terry-Luce model

TO DO: DOCUMENTATION!
"""

from rbtlModel_base import *



"""
Generalized BTL model

Parameters:
- n: number of students
- w: scores
- theta: parameters of function: scores --> abilities
- w_prior: prior for w (used to set regularization)
- lambda_a: regularization for a
"""
class rbtlModel(rbtlModel_base):
    
    def __init__(self, n):
        super(rbtlModel, self).__init__(n)
        self.theta = numpy.asarray([0.0,1.0]) # BTL

    # Set theta.
    def setTheta(self, new_theta):
        if len(new_theta) != 2:
            raise Exception('Incorrect theta length: [%s]' \
                            % ', '.join(map(str, new_theta)))
        self.theta = numpy.asarray(new_theta).astype(float)

    def getAbilities(self):
        g = numpy.zeros(self.n)
        for i in range(self.n):
            g[i] = self.theta[0] * self.w[i] + self.theta[1]
        return g

    # Grading ability of student i
    def getAbility(self, i):
        if i >= self.n:
            raise Exception("getAbility called on i = %d, but n = %d." % (i, self.n))
        return self.score2ability(self.w[i])

    # Compute ability g for the given score (using model's theta)
    def score2ability(self, score):
        return self.theta[0] * score + self.theta[1]

    # @param  g      grading abilities
    # @param  z_list  Data: z_list[k] = {(i,j,l)} indicating (i: j > l)
    # @return grad(theta, w)
    def gradients(self, g, z_list):
        grad_theta = numpy.zeros(len(self.theta))
        grad_w = numpy.zeros(self.n)
        a = self.theta[0]
        _w = self.w
        for k in range(len(z_list)):
            (i,j,l) = z_list[k] # i:j>l
            w_diff = _w[j] - _w[l]
            p_i_j_beats_l = P(g[i], _w[j], _w[l])
            tmp = w_diff * (1 - p_i_j_beats_l)
            grad_theta[0] += _w[i] * tmp
            grad_theta[1] += tmp
            grad_w[i] += a * tmp
            grad_w[j] += g[i] * (1 - p_i_j_beats_l)
            grad_w[l] -= g[i] * (1 - p_i_j_beats_l)
        return (grad_theta, grad_w)

    # @return grad(w[i], w[j], w[l]) for datum (i : j > l)
    def gradient_datum(self, g_i, i,j,l):
        P_i_j_l = P(g_i, self.w[j], self.w[l])
        grad_i = - self.theta[0] * (self.w[l] - self.w[j]) * (1 - P_i_j_l)
        grad_j = g_i * (1 - P_i_j_l)
        grad_l = - g_i * (1 - P_i_j_l)
        return (grad_i, grad_j, grad_l)

    # Count number of examples for which the more likely outcome occurs
    # (w.r.t. this model).
    def countCorrectExs(self, z_list):
        cnt = 0
        for k in range(len(z_list)):
            (i,j,l) = z_list[k] # (i:j>l)
            if self.w[j] > self.w[l]:
                cnt += 1
        return cnt

# END OF class rbtl_model(object)



# Try to load a model from filepath.
# If the file does not exist, create a model using testParams,
#  and save it to the filepath.
# @return model
def loadOrCreateModel(filepath, testParams):
    if os.path.isfile(filepath):
        try:
            fid = open(filepath, 'rb')
            model = pickle.load(fid)
            fid.close()
            print "Found model file: %s\n" % filepath
            return model
        except:
            print "Found model file CORRUPTED: %s\n" % filepath
            pass
    model = rbtl_model(testParams.n)
    model.genScores(testParams.scoreDist)
    model.theta = testParams.theta
    fid = open(filepath, 'wb')
    pickle.dump(model, fid)
    fid.close()
    return model


