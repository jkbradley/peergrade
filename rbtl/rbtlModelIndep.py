"""
Generalized Bradley-Terry-Luce model library

TO DO: DOCUMENTATION!

w,g independent
"""

from rbtlModel_base import *



"""
Generalized BTL model, with w and g decoupled

Parameters:
- n: number of students
- w: scores
- g: grading abilities
- w_prior: prior for w (used to set regularization)
- g_prior: prior for g (used to set regularization)
"""
class rbtlModelIndep(rbtlModel_base):
    
    def __init__(self, n):
        super(rbtlModelIndep, self).__init__(n)
        self.g = numpy.ones(n)

    def getAbilities(self):
        return self.g

    # Grading ability of student i
    def getAbility(self, i):
        if i >= self.n:
            raise Exception("getAbility called on i = %d, but n = %d." % (i, self.n))
        return self.g[i]
    
    # Generate grading abilities g.
    # @param  scoreDist  scoreDistribution type
    def genAbilities(self, scoreDist):
        self.g = numpy.zeros(self.n)
        for i in range(self.n):
            self.g[i] = scoreDist.genScore()

    # @param  z_list  Data: z_list[k] = {(i,j,l)} indicating (i: j > l)
    # @return [grad_w, grad_g]
    def gradients(self, z_list):
        grad_w = numpy.zeros(self.n)
        grad_g = numpy.zeros(self.n)
        _w = self.w
        _g = self.g
        for k in range(len(z_list)):
            (i,j,l) = z_list[k] # i:j>l
            p_i_l_beats_j = P(_g[i], _w[l], _w[j])
            grad_g[i] += (_w[j] - _w[l]) * p_i_l_beats_j
            grad_w[j] += _g[i] * p_i_l_beats_j
            grad_w[l] -= _g[i] * p_i_l_beats_j
        return (grad_w, grad_g)

# END OF class rbtlModelIndep



# Try to load a model from filepath.
# If the file does not exist, create a model using testParams,
#  and save it to the filepath.
# Args:
#   testParams: struct with fields: n, scoreDist_w, scoreDist_g
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
    model = rbtlModelIndep(testParams.n)
    model.genScores(testParams.scoreDist_w)
    model.genAbilities(testParams.scoreDist_g)
    fid = open(filepath, 'wb')
    pickle.dump(model, fid)
    fid.close()
    return model


