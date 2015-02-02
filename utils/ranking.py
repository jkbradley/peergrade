# Ranking utilities


# Compute ranking of a list.
# Ties are handled via fractional ranks:
#   If student i,i+1,...,j are ranked identically, they receive the rank (i+j)/2.
def fractionalRanking(vals):
    n = len(vals)
    tmp = []
    for i in range(n):
        tmp.append((vals[i],i))
    tmp = sorted(tmp)
    j_rank_list = []
    i = 0
    curval = tmp[0][0] - 1
    n_curvals = 0
    while i < n:
        (val,j) = tmp[i]
        if val == curval:
            n_curvals += 1
        else:
            frac_rank = i - (n_curvals - 1) / 2.0
            for k in range(n_curvals):
                j_rank_list.append((tmp[i-k-1][1],frac_rank))
            curval = val
            n_curvals = 1
        i += 1
    frac_rank = i - (n_curvals - 1) / 2.0
    for k in range(n_curvals):
        j_rank_list.append((tmp[i-k-1][1],frac_rank))
    ranks = [0]*n
    for (j,r) in j_rank_list:
        ranks[j] = r
    return ranks

    
