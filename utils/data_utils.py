# Utility functions for handling data.

import numpy, numpy.random, pickle

import index_builder

# Clean data by removing people who do not grade or were not graded.
#
# Arguments:
#   students   List of students, in index order (index used by z_list).
#   z_list     Comparison list: z_list[k] = [i,j,l] where i:j>l in comparison k.
#   requireGrader     If true, remove non-graders.
#   requireGradee      If true, remove non-gradees.
#
# @return [updated students,  updated z_list]
def clean_comparisons(students, z_list, requireGrader, requireGradee):
    orig_index = index_builder.index_builder()
    orig_index.add_all_items(students)
    # Find graders, gradees.
    grader_set = {}
    gradee_set = {}
    for z in z_list:
        i_id = orig_index.item(z[0])
        j_id = orig_index.item(z[1])
        l_id = orig_index.item(z[2])
        grader_set[i_id] = 1
        gradee_set[j_id] = 1
        gradee_set[l_id] = 1
    # Create new index,z_list.
    new_index = index_builder.index_builder()
    new_z_list = []
    for z in z_list:
        i_id = orig_index.item(z[0])
        j_id = orig_index.item(z[1])
        l_id = orig_index.item(z[2])
        if requireGrader and \
          ((j_id not in grader_set) or (l_id not in grader_set)):
            continue
        if requireGradee and i_id not in gradee_set:
            continue
        i = new_index.idx(i_id)
        j = new_index.idx(j_id)
        l = new_index.idx(l_id)
        new_z_list.append([i,j,l])
    return (new_index.item_list, new_z_list)


# Convert z_list to z_by_grader, where
#  z_list[k] = (i,j,l) where (i: j > l)
#  z_by_grader[i][j][l] = 1 if (i: j > l), and -1 o.w.
# @return z_by_grader
def convertZ_list2bygrader(z_list, n):
    z_by_grader = dict()
    for i in range(n):
        z_by_grader[i] = dict()
    for z_ in z_list:
        (i,j,l) = z_ # (i: j > l)
        if not (j in z_by_grader[i]):
            z_by_grader[i][j] = dict()
        if not (l in z_by_grader[i]):
            z_by_grader[i][l] = dict()
        z_by_grader[i][j][l] = 1
        z_by_grader[i][l][j] = -1
    return z_by_grader

# Given list of folds,
# Create train, test sets for fold i of n.
# Test set is i^th fold. Training set is the union of other folds.
#
# This cleans the set to only contain training set students.
#  (Other test sets students are removed.)
# This re-indexes the students.
#
# Arguments:
#   student_list   List of students, used to index z_list_folds.
#   z_list_folds   Output of splitFolds(), of length nfolds
#   i              i^th fold
# Returns:
#   (fold_students, fold_train_z_list, fold_test_z_list)
def getFoldi(student_list, z_list_folds, i, cleanComparisons=False):
    nfolds = len(z_list_folds)
    if i >= nfolds:
        raise Exception('getFoldi: i >= nfolds')
    fold_train_z_list = []
    for n in range(nfolds):
        if i != n:
            fold_train_z_list += z_list_folds[n]
    fold_test_z_list = z_list_folds[i]
    if cleanComparisons:
        [clean_students, clean_train_z_list] = \
          clean_comparisons(student_list, fold_train_z_list, \
                            requireGrader=requireGrader, requireGradee=requireGradee)
        clean_test_z_list = \
          reindex_z_list(fold_test_z_list, student_list, clean_students)
        return (clean_students, clean_train_z_list, clean_test_z_list)
    else:
        return (student_list, fold_train_z_list, fold_test_z_list)


# Partition into folds, dividing each grader's comparisons evenly.
# Arguments:
#   z_list    Comparison list: z_list[k] = (i,j,l) where (i: j > l)
#   n         Number of graders.  Graders must be indexed in z_list by {0,...,n-1}
#   nfolds    Number of folds for CV.
#
# @return array of z_list for each fold
def splitFolds(z_list, n, nfolds):
    z_by_grader = convertZ_list2bygrader(z_list, n)
    z_list_folds = []
    for k in range(nfolds):
        z_list_folds.append([])
    grader_perm = numpy.random.permutation(n)
    fold_i = 0 # next fold in z_list_folds to add a grade to
    for i in grader_perm:
        z_list_i = [] # grades made by grader i
        for j in z_by_grader[i].iterkeys():
            for l in z_by_grader[i][j].iterkeys():
                if (z_by_grader[i][j][l] == 1):
                    z_list_i.append([i,j,l])
        perm = numpy.random.permutation(len(z_list_i))
        for z_ind in range(len(z_list_i)):
            z = z_list_i[perm[z_ind]]
            z_list_folds[fold_i].append(z)
            fold_i += 1
            if fold_i >= nfolds:
                fold_i = 0
    return z_list_folds

# @param z_list  [(i,j,grade)]
# @return array of z_list for each fold
def splitFoldsByGrade(z_list, n, nfolds):
    z_by_grader = convert_grade_list2bygrader(z_list, n)
    z_list_folds = []
    for k in range(nfolds):
        z_list_folds.append([])
    for i in range(n):
        z_list_i = [] # grades made by grader i
        for j in z_by_grader[i].iterkeys():
            z_list_i.append([i,j,z_by_grader[i][j]])
        while (len(z_list_i) % nfolds != 0):
            z_list_i.append([]) # Evenly distribute grades among folds
        perm = numpy.random.permutation(len(z_list_i))
        k = 0 # fold
        for z_ind in perm:
            z = z_list_i[z_ind]
            if z != []:
                z_list_folds[k].append(z)
            k += 1
            if k >= nfolds:
                k = 0
    print 'Fold sizes:'
    for k in range(nfolds):
        print '%d' % len(z_list_folds[k])
    return z_list_folds

# Convert grade_list to grade_by_gradee.
#
# @param grade_list[k] = (i,j,grade) where (i grades j)
# @return grade_by_gradee[i][j] = grade j gave to i
def convert_grade_list2bygradee(grade_list, n):
    grade_by_gradee = dict()
    for i in range(n):
        grade_by_gradee[i] = dict()
    for z_ in grade_list:
        (i,j,grade) = z_ # (i grades j)
        grade_by_gradee[j][i] = grade
    return grade_by_gradee

# Convert grade_list to grade_by_grader.
#
# @param grade_list[k] = (i,j,grade) where (i grades j)
# @return grade_by_grader[i][j] = grade i gave to j
def convert_grade_list2bygrader(grade_list, n):
    grade_by_grader = dict()
    for i in range(n):
        grade_by_grader[i] = dict()
    for z_ in grade_list:
        (i,j,grade) = z_ # (i grades j)
        grade_by_grader[i][j] = grade
    return grade_by_grader

# Convert z_by_grader to z_list, where
#  z_by_grader[i][j][l] = 1 if (i: j > l), and -1 o.w.
#  z_list[k] = (i,j,l) where (i: j > l)
# @return z_list
def convertZ_grader2list(z_by_grader):
    n = len(z_by_grader)
    k = 0
    z_list = []
    for i in range(n):
        for j in z_by_grader[i].iterkeys():
            for l in z_by_grader[i][j].iterkeys():
                if (z_by_grader[i][j][l] == 1):
                    z_list.append([i,j,l])
    return z_list

# Convert z_list to z_by_grader, where
#  z_list[k] = (i,j,l) where (i: j > l)
#  z_by_grader[i][j][l] = 1 if (i: j > l), and -1 o.w.
# @return z_by_grader
def convertZ_list2bygrader(z_list, n):
    z_by_grader = dict()
    for i in range(n):
        z_by_grader[i] = dict()
    for (i,j,l) in z_list: # (i: j > l)
        if not (j in z_by_grader[i]):
            z_by_grader[i][j] = dict()
        if not (l in z_by_grader[i]):
            z_by_grader[i][l] = dict()
        z_by_grader[i][j][l] = 1
        z_by_grader[i][l][j] = -1
    return z_by_grader


# Remove students not in new_students.
# Re-indexes students according to new_students list.
#
# Arguments:
#   orig_z_list    [[i,j,l]]
#   orig_students  [list of student IDs]  (indexes orig_z_list)
#   new_students   [sub-list of student IDs]  (indexes returned z_list)
def reindex_z_list(orig_z_list, orig_students, new_students):
    new_index = index_builder.index_builder()
    new_index.add_all_items(new_students)
    new_z_list = []
    for z in orig_z_list:
        i = orig_students[z[0]]
        j = orig_students[z[1]]
        l = orig_students[z[2]]
        if i in new_index and j in new_index and l in new_index:
            new_z_list.append([new_index.idx(i), new_index.idx(j), new_index.idx(l)])
    return new_z_list

# Load comparisons from pickle file in standard format.
# Pickle file should contain 3 items:
#   [student IDs]
#   [training z_list]
#   [test z_list] (optional)
#
# @return (student_list, train_z_list, test_z_list)
#   If no test_z_list, return test_z_list=None.
def load_comparisons_pickle(datapath, verbose = False):
    fid = open(datapath, 'rb')
    student_list = pickle.load(fid)
    train_z_list = pickle.load(fid)
    try:
        test_z_list = pickle.load(fid)
    except:
        test_z_list = None
    fid.close()
    if verbose:
        if test_z_list is not None:
            ntest = len(test_z_list)
        else:
            ntest = 0
        print 'Loaded %d students, %d training comparisons, %d test comparisons' \
          % (len(student_list), len(train_z_list), ntest)
    return (student_list, train_z_list, test_z_list)

# Remove students from data.
# Re-index.
def removeStudents(orig_z_list, orig_student_list, remove_students):
    rset = {}
    for r in remove_students:
        rset[r] = 0
    z_list = []
    new_student_index = index_builder.index_builder()
    rcount = 0
    for z in orig_z_list:
        (i,j,l) = (orig_student_list[z[0]], orig_student_list[z[1]], \
                   orig_student_list[z[2]])
        if (i in rset) or (j in rset) or (l in rset):
            rcount += 1
            continue
        newi = new_student_index.idx(i)
        newj = new_student_index.idx(j)
        newl = new_student_index.idx(l)
        z_list.append([newi,newj,newl])
    return (new_student_index.item_list, z_list, rcount)

