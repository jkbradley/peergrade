import random

# @todo Modify to avoid duplicate (overwritten) measurements (i,r1,r2)
#
# @param w  Scores
# @param g  Grading abilities
# @param k  Total number of grades
# @return z_by_grader  z_by_grader[i][j][l] = 1 if (i: j > l), and -1 o.w.
def genGrades(w, g, k):
    n = len(w)
    z = dict()
    for i in range(n):
        z[i] = dict()
        k_i = k / n
        if (i+1 == n):
            k_i = k - (n-1) * (k/n)
        for k_ in range(k_i):
            r1 = random.randint(0,n-1) # j
            while r1==i:
                r1 = random.randint(0,n-1)
            r2 = random.randint(0,n-1) # l
            while (r2==i) or (r2==r1):
                r2 = random.randint(0,n-1)
            if not (r1 in z[i]):
                z[i][r1] = dict()
            if not (r2 in z[i]):
                z[i][r2] = dict()
            rand = random.random()
            p_i_r1_beats_r2 = P(g[i], w[r1], w[r2])
            if rand < p_i_r1_beats_r2:   #r1 wins
                z[i][r1][r2] = 1
                z[i][r2][r1] = -1
            else:                        #r2 wins
                z[i][r1][r2] = -1
                z[i][r2][r1] = 1
    return z
