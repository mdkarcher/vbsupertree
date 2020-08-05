from importlib import reload
# from collections import OrderedDict
# from more_itertools import powerset

# from small_trees import *
import random
import classes
reload(classes)

from classes import *


# Gradient descent experiments

reload(classes)
from classes import *


X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")

ccd = CCDSet.random(X)
ccd_small = CCDSet.random(Xbar)

candidate, kl_list = ccd_gradient_descent_one(starting=ccd, reference=ccd_small, starting_gamma=1.0, max_iteration=200)

candidate_tree_dist = candidate.tree_distribution().restrict(Xbar)
candidate_tree_dist2 = candidate.restrict(Xbar).tree_distribution()
reference_tree_dist = ccd_small.tree_distribution()

reference_tree_dist.kl_divergence(candidate_tree_dist)
reference_tree_dist.kl_divergence(candidate_tree_dist2)

## Multiple reference distributions

X = Clade("ABCDEFG")
restrictions = [Clade("ABCDE"), Clade("CDEFG")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd1)

ccd, kl_list = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

true_ccd.kl_divergence(ccd)
ccd.kl_divergence(true_ccd)

true_tree = true_ccd.tree_distribution()
est_tree = ccd.tree_distribution()
true_tree.kl_divergence(est_tree)

references[0].kl_divergence(ccd.restrict(restrictions[0]))
references[1].kl_divergence(ccd.restrict(restrictions[1]))

## Multiple disagreeing reference distributions

X = Clade("ABCDEFG")
restrictions = [Clade("ABCDEF"), Clade("BCDEFG")]

# No true SBN to match
references = [CCDSet.random(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
references[0].kl_divergence(starting_ccd1.restrict(restrictions[0]))
references[1].kl_divergence(starting_ccd1.restrict(restrictions[1]))

ccd, kl_list = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

true_ccd.kl_divergence(ccd)
ccd.kl_divergence(true_ccd)

true_tree = true_ccd.tree_distribution()
est_tree = ccd.tree_distribution()
true_tree.kl_divergence(est_tree)

references[0].kl_divergence(ccd.restrict(restrictions[0]))
references[1].kl_divergence(ccd.restrict(restrictions[1]))

# Different starting points experiment

X = Clade("ABCDEFG")
restrictions = [Clade("ABCDE"), Clade("CDEFG")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd1)

ccd1, kl_list1 = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

true_ccd.kl_divergence(ccd1)
ccd1.kl_divergence(true_ccd)

references[0].kl_divergence(ccd1.restrict(restrictions[0]))
references[1].kl_divergence(ccd1.restrict(restrictions[1]))

starting_ccd2 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd2)
starting_ccd1.kl_divergence(starting_ccd2)

ccd2, kl_list2 = ccd_gradient_descent(starting=starting_ccd2, references=references, starting_gamma=2.0, max_iteration=200)

true_ccd.kl_divergence(ccd2)
ccd2.kl_divergence(true_ccd)
ccd1.kl_divergence(ccd2)

references[0].kl_divergence(ccd2.restrict(restrictions[0]))
references[1].kl_divergence(ccd2.restrict(restrictions[1]))

# Low ambiguity, multiple combination

X = Clade("ABCD")
restrictions = [Clade("BCD"), Clade("ACD"), Clade("ABD"), Clade("ABC")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd1)

ccd, kl_list = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

true_ccd.kl_divergence(ccd)
ccd.kl_divergence(true_ccd)

true_tree = true_ccd.tree_distribution()
est_tree = ccd.tree_distribution()
true_tree.kl_divergence(est_tree)

[reference.kl_divergence(ccd.restrict(restriction)) for (reference, restriction) in zip(references, restrictions)]
# references[0].kl_divergence(ccd.restrict(restrictions[0]))

# Degrees of freedom experiment

reload(classes)
from classes import *


X = Clade("ABCDEFG")
Xbar = Clade("ABCDEF")

ccd = CCDSet.random(X)
ccd_small = CCDSet.random(Xbar)

ccd.degrees_of_freedom()
ccd_small.degrees_of_freedom()

len(ccd)
len(ccd_small)



# Non-dense Support experiment

reload(classes)
from classes import *


taxon_set1 = Clade("ABCDE")
taxon_set2 = Clade("CDEFG")

ccd1 = CCDSet.random(taxon_set1, concentration=0.3, cutoff=0.1)
len(ccd1)
len(ccd1.tree_distribution())
ccd2 = CCDSet.random(taxon_set2, concentration=0.3, cutoff=0.1)
len(ccd2)
len(ccd2.tree_distribution())

support1 = ccd1.support()
len(support1)
support2 = ccd2.support()
len(support2)

mutual_support = support1.mutualize(support2)
len(mutual_support)
len(mutual_support.all_trees())

mutual_starting_ccd = CCDSet.random_from_support(mutual_support)
len(mutual_starting_ccd)
len(mutual_starting_ccd.tree_distribution())

# True reference test

from importlib import reload
import numpy as np
import classes
reload(classes)
from classes import *


X = Clade("ABCDEFG")
restrictions = [Clade("ABCDE"), Clade("CDEFG"), Clade("ABDFG")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd)

[reference.kl_divergence(starting_ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

ccd, kl_list, true_kl_list = ccd_gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

np.round(np.array(true_kl_list), 3)

kl_list[0] - kl_list[-1]
true_kl_list[0] - true_kl_list[-1]

true_ccd.kl_divergence(ccd)
ccd.kl_divergence(true_ccd)

true_tree = true_ccd.tree_distribution()
est_tree = ccd.tree_distribution()
true_tree.kl_divergence(est_tree)

# Checking for ~zero derivatives

reload(classes)
from classes import *


X = Clade("ABCDEF")
restrictions = [Clade("ABCD"), Clade("ABEF"), Clade("CDEF")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd1)
split_kl = true_ccd.kl_divergence_ccd_verbose(starting_ccd1)
sum(split_kl.values())
[reference.kl_divergence(starting_ccd1.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

organized_gradient = starting_ccd1.restricted_kl_divergence_gradient_multi_organized(references)
l2norm_dict_of_dicts(organized_gradient)

ccd, kl_list, true_kl_list = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

true_ccd.kl_divergence(ccd)
kl_list[0] - kl_list[-1]
true_kl_list[0] - true_kl_list[-1]

split_kl_after = true_ccd.kl_divergence_ccd_verbose(ccd)
[reference.kl_divergence(ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]
[reference.kl_divergence(true_ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

for clade in split_kl:
    print(f"Clade {str(clade)}: before={split_kl[clade]:8.4g}, after={split_kl_after[clade]:8.4g}")

## Checking for ~zero derivatives and degrees of freedom test

X = Clade("ABCDEFG")
restrictions = [Clade("ABCDEF"), Clade("ABCDEG"), Clade("ABCDFG"), Clade("ABCEFG"), Clade("ABDEFG"), Clade("ACDEFG"), Clade("BCDEFG")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd1)
split_kl = true_ccd.kl_divergence_ccd_verbose(starting_ccd1)
sum(split_kl.values())
[reference.kl_divergence(starting_ccd1.restrict(restriction)) for reference, restriction in zip(references, restrictions)]
sum(reference.kl_divergence(starting_ccd1.restrict(restriction)) for reference, restriction in zip(references, restrictions))

organized_gradient = starting_ccd1.restricted_kl_divergence_gradient_multi_organized(references)
l2norm_dict_of_dicts(organized_gradient)

ccd, kl_list, true_kl_list = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=500)

true_ccd.kl_divergence(ccd)
kl_list[0] - kl_list[-1]
true_kl_list[0] - true_kl_list[-1]

split_kl_after = true_ccd.kl_divergence_ccd_verbose(ccd)
[reference.kl_divergence(ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]
[reference.kl_divergence(true_ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

[reference.degrees_of_freedom() for reference in references]
sum(reference.degrees_of_freedom() for reference in references)
true_ccd.degrees_of_freedom()

# Multiple start points experiment

reload(classes)
from classes import *


X = Clade("ABCDEF")
restrictions = [Clade("ABCDE"), Clade("ABCEF"), Clade("ACDEF")]
# restrictions = [Clade("ABCDE"), Clade("ABCDF"), Clade("ABCEF"), Clade("ABDEF"), Clade("ACDEF")]

true_ccd = CCDSet.random(X)
references = [true_ccd.restrict(restriction) for restriction in restrictions]

starting_ccd1 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd1)
split_kl1 = true_ccd.kl_divergence_ccd_verbose(starting_ccd1)
split_kl1
sum(split_kl1.values())
[reference.kl_divergence(starting_ccd1.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

organized_gradient1 = starting_ccd1.restricted_kl_divergence_gradient_multi_organized(references)
l2norm_dict_of_dicts(organized_gradient1)

ccd1, kl_list1, true_kl_list1 = ccd_gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

true_ccd.kl_divergence(ccd1)
kl_list1[0] - kl_list1[-1]
true_kl_list1[0] - true_kl_list1[-1]

split_kl_after1 = true_ccd.kl_divergence_ccd_verbose(ccd1)
split_kl_after1
[reference.kl_divergence(ccd1.restrict(restriction)) for reference, restriction in zip(references, restrictions)]
[reference.kl_divergence(true_ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

for clade in split_kl1:
    print(f"Clade {str(clade)}: before={split_kl1[clade]:8.4g}, after={split_kl_after1[clade]:8.4g}")

starting_ccd2 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd2)
split_kl2 = true_ccd.kl_divergence_ccd_verbose(starting_ccd2)
split_kl2
sum(split_kl2.values())
[reference.kl_divergence(starting_ccd2.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

organized_gradient2 = starting_ccd2.restricted_kl_divergence_gradient_multi_organized(references)
l2norm_dict_of_dicts(organized_gradient2)

ccd2, kl_list2, true_kl_list2 = ccd_gradient_descent(starting=starting_ccd2, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

true_ccd.kl_divergence(ccd2)
kl_list2[0] - kl_list2[-1]
true_kl_list2[0] - true_kl_list2[-1]

split_kl_after2 = true_ccd.kl_divergence_ccd_verbose(ccd2)
split_kl_after2
[reference.kl_divergence(ccd2.restrict(restriction)) for reference, restriction in zip(references, restrictions)]
[reference.kl_divergence(true_ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

for clade in split_kl2:
    print(f"Clade {str(clade)}: before={split_kl2[clade]:8.4g}, after={split_kl_after2[clade]:8.4g}")

mutual_split_kl = starting_ccd1.kl_divergence_ccd_verbose(starting_ccd2)
mutual_split_kl
starting_ccd1.kl_divergence(starting_ccd2)
mutual_split_kl_after = ccd1.kl_divergence_ccd_verbose(ccd2)
mutual_split_kl_after
ccd1.kl_divergence(ccd2)

for clade in split_kl1:
    if split_kl1[clade] == 0.0:
        continue
    print(f"Clade {str(clade)}:")
    print(f"  Run 1: before={split_kl1[clade]:8.4g}, after={split_kl_after1[clade]:8.4g}, %remaining={100*split_kl_after1[clade]/split_kl1[clade]}%")
    print(f"  Run 2: before={split_kl2[clade]:8.4g}, after={split_kl_after2[clade]:8.4g}, %remaining={100*split_kl_after2[clade]/split_kl2[clade]}%")
    print(f"  Mutual: before={mutual_split_kl[clade]:8.4g}, after={mutual_split_kl_after[clade]:8.4g}, %remaining={100*mutual_split_kl_after[clade]/mutual_split_kl[clade]}%")


# Sparsity experiment

from importlib import reload
# import numpy as np
import classes

reload(classes)
from classes import *


X = Clade("ABCDEFG")
restrictions = [Clade("ABCDE"), Clade("CDEFG"), Clade("ABDFG")]

true_ccd = CCDSet.random(X)
# true_ccd = CCDSet.random_with_sparsity(X, sparsity=0.75)
len(true_ccd)

references = [true_ccd.restrict(restriction) for restriction in restrictions]

common_support = None
for reference in references:
    if common_support is None:
        common_support = reference.support(include_trivial=True)
        continue
    common_support = common_support.mutualize(reference.support(include_trivial=True))
len(common_support)

common_support.is_complete()

starting_ccd = CCDSet.random_from_support(common_support)
len(starting_ccd)

true_ccd.kl_divergence(starting_ccd)

[reference.kl_divergence(starting_ccd.restrict(restriction)) for reference, restriction in zip(references, restrictions)]

ccd, kl_list, true_kl_list = ccd_gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

np.round(np.array(true_kl_list), 3)

kl_list[0] - kl_list[-1]
true_kl_list[0] - true_kl_list[-1]

true_ccd.kl_divergence(ccd)
ccd.kl_divergence(true_ccd)

true_tree = true_ccd.tree_distribution()
est_tree = ccd.tree_distribution()
true_tree.kl_divergence(est_tree)


# cross_multiple experiments

reload(classes)
from classes import *

# subsplits = [Subsplit("A", "B"), Subsplit("C", "D"), Subsplit("E", "F"), Subsplit("G", "H")]
subsplits = [Subsplit("A", "BC"), Subsplit("CE", "D"), Subsplit("E", "F"), Subsplit("G", "H")]

rearranged = random.sample(subsplits, len(subsplits))
list1 = Subsplit.cross_multiple(rearranged)
list2 = Subsplit.cross_multiple_naive(rearranged)
set1 = set(list1)
set2 = set(list2)
set1 == set2

len(list1)
len(set1)
len(list2)
len(set2)

print(list1)

print(set1)
print(set2)

# common_support experiments

reload(classes)
from classes import *

X = "ABCDEF"
# restrictions = [Clade("ABCDE"), Clade("ABCEF"), Clade("ACDEF")]
restrictions = [Clade("ABCDE"), Clade("ABCDF"), Clade("ABCEF"), Clade("ABDEF"), Clade("ACDEF"), Clade("BCDEF")]

ccd = CCDSet.random_with_sparsity(X, 0.75)
len(ccd)

support = ccd.support()
len(support)
len(ccd.restrict(restrictions[0]))
len(support.restrict(restrictions[0]))

reference_supports = [support.restrict(restriction) for restriction in restrictions]
common_support = SubsplitSupport.common_support(reference_supports)
len(common_support)

common_support_1by1 = reference_supports[0]
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[1])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[2])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[3])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[4])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[5])
len(common_support_1by1)

support.to_set().issubset(common_support_1by1.to_set())
support.to_set().issubset(common_support.to_set())
common_support.to_set() == common_support_1by1.to_set()

#

reload(classes)
from classes import *

# X = "ABCDEF"
# restrictions = [Clade("ABCDE"), Clade("ABCDF"), Clade("ABCEF"), Clade("ABDEF"), Clade("ACDEF"), Clade("BCDEF")]
X = Clade("ABCDEFG")
restrictions = [Clade("ABCDEF"), Clade("ABCDEG"), Clade("ABCDFG"), Clade("ABCEFG"), Clade("ABDEFG"), Clade("ACDEFG"), Clade("BCDEFG")]

true_ccd = CCDSet.random_with_sparsity(X, 0.75)
len(true_ccd)

true_support = true_ccd.support()
len(true_support)
len(true_ccd.restrict(restrictions[0]))
len(true_support.restrict(restrictions[0]))

references = [true_ccd.restrict(restriction) for restriction in restrictions]
reference_supports = [true_support.restrict(restriction) for restriction in restrictions]

common_support_1by1 = reference_supports[0]
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[1])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[2])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[3])
len(common_support_1by1)

selected_support = common_support_1by1

common_support_1by1 = common_support_1by1.mutualize(reference_supports[4])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[5])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[6])
len(common_support_1by1)

true_support.to_set().issubset(common_support_1by1.to_set())
true_support.to_set() == common_support_1by1.to_set()

starting_ccd = CCDSet.random_from_support(common_support_1by1)
true_ccd.kl_divergence(starting_ccd)
ccd_all, kl_list_all, true_kl_list_all = ccd_gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
true_ccd.kl_divergence(ccd_all)

len(selected_support)
starting_ccd_some = CCDSet.random_from_support(selected_support)
true_ccd.kl_divergence(starting_ccd_some)
ccd_some, kl_list_some, true_kl_list_some = ccd_gradient_descent(starting=starting_ccd_some, references=references[:4], starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
true_ccd.kl_divergence(ccd_some)

#

reload(classes)
from classes import *


X = Clade("ABCDEFGH")
restrictions = [Clade("ABCDEFG"), Clade("ABCDEGH"), Clade("ABCEFGH"), Clade("ACDEFGH")]

true_ccd = CCDSet.random_with_sparsity(X, 0.8)
len(true_ccd)

true_support = true_ccd.support()
len(true_support)
len(true_ccd.restrict(restrictions[0]))
len(true_support.restrict(restrictions[0]))

references = [true_ccd.restrict(restriction) for restriction in restrictions]
reference_supports = [true_support.restrict(restriction) for restriction in restrictions]

common_support_1by1 = reference_supports[0]
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[1])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[2])
len(common_support_1by1)
common_support_1by1 = common_support_1by1.mutualize(reference_supports[3])
len(common_support_1by1)

# selected_support = common_support_1by1
# common_support_1by1 = common_support_1by1.mutualize(reference_supports[4])
# len(common_support_1by1)
# common_support_1by1 = common_support_1by1.mutualize(reference_supports[5])
# len(common_support_1by1)
# common_support_1by1 = common_support_1by1.mutualize(reference_supports[6])
# len(common_support_1by1)

true_support.to_set().issubset(common_support_1by1.to_set())
true_support.to_set() == common_support_1by1.to_set()

starting_ccd = CCDSet.random_from_support(common_support_1by1)
true_ccd.kl_divergence(starting_ccd)
ccd_all, kl_list_all, true_kl_list_all = ccd_gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
true_ccd.kl_divergence(ccd_all)

#

reload(classes)
from classes import *


X = Clade("ABCDEFGH")
ambig = 2
n_references = 5
restrictions = list(set(list(map(Clade, (random.sample(X, len(X)-ambig) for _ in range(n_references))))))
Clade(set.union(*map(set, restrictions))) == X
len(restrictions)

true_ccd = CCDSet.random_with_sparsity(X, 0.8)
len(true_ccd)

true_support = true_ccd.support()
len(true_support)
true_support.is_complete()

references = [true_ccd.restrict(restriction) for restriction in restrictions]
reference_supports = [true_support.restrict(restriction) for restriction in restrictions]

common_support_1by1 = reference_supports[0]
print(len(common_support_1by1))
for ref_sup in reference_supports[1:]:
    common_support_1by1 = common_support_1by1.mutualize(ref_sup)
    print(len(common_support_1by1))
    print(common_support_1by1.is_complete())
common_support_1by1.is_complete(verbose=True)

common_support = SubsplitSupport.common_support(reference_supports)

true_support.to_set().issubset(common_support_1by1.to_set())
true_support.to_set() == common_support_1by1.to_set()

starting_ccd = CCDSet.random_from_support(common_support_1by1)
true_ccd.kl_divergence(starting_ccd)
ccd_all, kl_list_all, true_kl_list_all = ccd_gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
true_ccd.kl_divergence(ccd_all)

