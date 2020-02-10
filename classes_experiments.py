from importlib import reload
from collections import OrderedDict
from more_itertools import powerset

from small_trees import *

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

candidate, kl_list = gradient_descent_one(starting=ccd, reference=ccd_small, starting_gamma=1.0, max_iteration=200)

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

ccd, kl_list = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

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

ccd, kl_list = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

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

ccd1, kl_list1 = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

true_ccd.kl_divergence(ccd1)
ccd1.kl_divergence(true_ccd)

references[0].kl_divergence(ccd1.restrict(restrictions[0]))
references[1].kl_divergence(ccd1.restrict(restrictions[1]))

starting_ccd2 = CCDSet.random(X)
true_ccd.kl_divergence(starting_ccd2)
starting_ccd1.kl_divergence(starting_ccd2)

ccd2, kl_list2 = gradient_descent(starting=starting_ccd2, references=references, starting_gamma=2.0, max_iteration=200)

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

ccd, kl_list = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, max_iteration=200)

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

# PCMiniSet experiments

from collections import Counter

reload(classes)
from classes import *

pcm = PCMiniSet(Subsplit("ABC", "DEF"))
pcm.add(Subsplit("AB", "C"), 0.5)
pcm.add(Subsplit("A", "BC"), 0.5)
pcm.add(Subsplit("D", "EF"), 0.2)
pcm.add(Subsplit("DE", "F"), 0.8)
pcm.check()

pcm.normalize()
pcm.left.check()
pcm.right.check()
pcm.check()

pcm.sample()
pcm.left.sample()

left_sample = [pcm.left.sample() for _ in range(100)]
Counter(left_sample)

right_sample = [pcm.right.sample() for _ in range(100)]
Counter(right_sample)

Counter(pcm.right.sample() for _ in range(100))

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

ccd, kl_list, true_kl_list = gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

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

ccd, kl_list, true_kl_list = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

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

ccd, kl_list, true_kl_list = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=500)

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

ccd1, kl_list1, true_kl_list1 = gradient_descent(starting=starting_ccd1, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

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

ccd2, kl_list2, true_kl_list2 = gradient_descent(starting=starting_ccd2, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

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

ccd, kl_list, true_kl_list = gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)

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
ccd_all, kl_list_all, true_kl_list_all = gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
true_ccd.kl_divergence(ccd_all)

len(selected_support)
starting_ccd_some = CCDSet.random_from_support(selected_support)
true_ccd.kl_divergence(starting_ccd_some)
ccd_some, kl_list_some, true_kl_list_some = gradient_descent(starting=starting_ccd_some, references=references[:4], starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
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
ccd_all, kl_list_all, true_kl_list_all = gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
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
ccd_all, kl_list_all, true_kl_list_all = gradient_descent(starting=starting_ccd, references=references, starting_gamma=2.0, true_reference=true_ccd, max_iteration=200)
true_ccd.kl_divergence(ccd_all)

# PCSS experiments

from importlib import reload
import classes

reload(classes)
from classes import *

tree = MyTree.random("ABCDE")
print(tree)
list(tree.traverse_subsplits())
list(tree.traverse_pcsss())

# Restriction experiment for unit tests

reload(classes)
from classes import *

X = Clade("ABCDEF")
restriction = Clade("ABCDE")

ccd = CCDSet.random_with_sparsity(X, sparsity=0.7)
dist = ccd.tree_distribution()
dist_res = dist.restrict(restriction)

res_ccd = ccd.restrict(restriction)
res_ccd_dist = res_ccd.tree_distribution()

dist_res.kl_divergence(res_ccd_dist)
res_ccd_dist.kl_divergence(dist_res)

res_ccd_dist_ccd = CCDSet.from_tree_distribution(res_ccd_dist)

res_ccd.kl_divergence(res_ccd_dist_ccd)
res_ccd_dist_ccd.kl_divergence(res_ccd)

dist_res_ccd = CCDSet.from_tree_distribution(dist_res)

dist_res_ccd.kl_divergence(res_ccd)
res_ccd.kl_divergence(dist_res_ccd)

dist_res_ccd_dist = dist_res_ccd.tree_distribution()

res_ccd_dist.kl_divergence(dist_res_ccd_dist)
dist_res_ccd_dist.kl_divergence(res_ccd_dist)

# PCSS initial experiments (for potential unit tests)

reload(classes)
from classes import *

pcss_conditional_probs = {
    PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
    PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
    PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2/3,
    PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1/3,
    PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
    PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
}
scd = SCDSet(pcss_conditional_probs)
scd.normalize()
result = scd.support()
# result = PCSSSupport()
# result.add(PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")))
# result.update(set(scd.support()))
# for pcss in scd.iter_pcss(): result.add(pcss)
result
result.is_complete(verbose=True)
scd.check_distributions()
scd.check()
tree = MyTree("((A,(B,C)),D);")
list(tree.traverse_pcsss())
scd.log_likelihood(tree)
math.log(0.2)

# PCMiniSet random experiments

parent = Subsplit("ABCD", "EFG")
pcmini = PCMiniSet.random(parent)
pcmini

pcmini2 = PCMiniSet.random_with_sparsity(parent, 0.5)
pcmini2

scd = SCDSet.random("ABCD")
scd
len(scd)
scd.check()

tree = scd.random_tree()
print(tree)

big_scd = SCDSet.random("ABCDEFG")
big_tree = big_scd.random_tree()
print(big_tree)

# SCD tree_distribution

reload(classes)
from classes import *

pcss_conditional_probs = {
    PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
    PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
    PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2/3,
    PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1/3,
    PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
    PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
}
scd = SCDSet(pcss_conditional_probs)
scd.normalize()

tree_dist = scd.tree_distribution(verbose=True)
print(tree_dist)

tree = MyTree("((A,(B,C)),D);")
print(tree)

scd.log_likelihood(tree)
tree_dist.get_log(tree)

# SCD random tree_distribution

reload(classes)
from classes import *

scd = SCDSet.random("ABCDE")
tree_dist = scd.tree_distribution()

len(scd)
len(tree_dist)

abs_diffs = []
rel_diffs = []
for tree in tree_dist:
    scd_lik = scd.likelihood(tree)
    tre_lik = tree_dist.get(tree)
    abs_diff = abs(scd_lik - tre_lik)
    rel_diff = abs_diff/abs(tre_lik)
    print(f"Abs diff: {abs_diff:8.4g}, Rel diff: {rel_diff:8.4g}")
    abs_diffs.append(abs_diff)
    rel_diffs.append(rel_diff)

# SCD test parent_probabilities

import random

reload(classes)
from classes import *

scd = SCDSet.random("ABCDE")
tree_dist = scd.tree_distribution()
parent_probs = scd.subsplit_probabilities()

parent, prob = random.choice(list(parent_probs.items()))

tree_dist.feature_prob(parent)
prob

# Double check CCD probability functions
# Result: iter_ functions have multiple entries per each subsplit

reload(classes)
from classes import *

ccd = CCDSet.random("ABCDEF")

uncond = ccd.unconditional_probabilities()
for subsplit, joint_prob in ccd.iter_unconditional_probabilities():
    print(f"Subsplit {subsplit}: {uncond[subsplit]:8.4g} {joint_prob:8.4g}")

len(uncond)
len(list(ccd.iter_unconditional_probabilities()))

# pcss_probabilities experiment

import random

reload(classes)
from classes import *

scd = SCDSet.random("ABCDE")
tree_dist = scd.tree_distribution()
pcss_probs = scd.pcss_probabilities()

pcss, prob = random.choice(list(pcss_probs.items()))

tree_dist.prob_all([pcss.parent, pcss.child])
prob

# SCDSet kl implementations

reload(classes)
from classes import *

pcss_conditional_probs1 = {
    PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
    PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
    PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
    PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
    PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
    PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
}
pcss_conditional_probs2 = {
    PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.5,
    PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.5,
    PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 0.5,
    PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 0.5,
    PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
    PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
}
scd1 = SCDSet(pcss_conditional_probs1)
scd1.normalize()
dist1 = scd1.tree_distribution()
scd2 = SCDSet(pcss_conditional_probs2)
scd2.normalize()
dist2 = scd2.tree_distribution()

dist1.kl_divergence(dist2)  # 0.05411532090976828
scd1.kl_divergence(scd2)  # 0.05411532090976828
dist1.kl_divergence(scd2)  # 0.05411532090976828
scd1.kl_divergence(dist2)  # 0.05411532090976828

# random SCD kl implementation

taxa = Clade("ABCDE")
ccd1 = CCDSet.random(taxa)
ccd2 = CCDSet.random(taxa)
dist1 = ccd1.tree_distribution()
dist2 = ccd2.tree_distribution()
dist_dist = dist1.kl_divergence(dist2)
ccd_dist = ccd1.kl_divergence(dist2)
ccd_ccd = ccd1.kl_divergence(ccd2)
dist_ccd = dist1.kl_divergence(ccd2)

dist_dist, ccd_dist
ccd_dist, ccd_ccd
ccd_ccd, dist_ccd
dist_ccd, dist_dist

# SCD restrict

reload(classes)
from classes import *

pcss_conditional_probs = {
    PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
    PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
    PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2/3,
    PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1/3,
    PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
    PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
    PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
}
scd = SCDSet(pcss_conditional_probs)
scd.normalize()

restriction = set("ABC")
restricted_scd = scd.restrict(restriction)

# SCD subsplit_to_subsplit

reload(classes)
from classes import *

root_clade = Clade("ABCDE")
root_subsplit = Subsplit(root_clade)
scd = SCDSet.random(root_clade)
tree_dist = scd.tree_distribution()

s2s_probs = scd.subsplit_to_subsplit_probabilities()
s_probs = scd.subsplit_probabilities()

parent = Subsplit(Clade("ABD"), Clade("C"))
s_probs[parent], s2s_probs[parent][root_subsplit]
s2s_probs[parent][parent], 1.0

child = Subsplit(Clade("A"), Clade("BD"))
den = tree_dist.feature_prob(parent)
num = tree_dist.prob_all([parent, child])
s2s_probs[child][parent], num / den

# SCD restrict

reload(classes)
from classes import *

root_clade = Clade("ABCD")
restriction = Clade("ABC")

scd = SCDSet.random_with_sparsity(root_clade, sparsity=0.0)
dist = scd.tree_distribution()
dist_res = dist.restrict(restriction)

scd_res = scd.restrict(restriction, verbose=True)
scd_res_dist = scd_res.tree_distribution()

scd_res_dist_scd = SCDSet.from_tree_distribution(scd_res_dist)

scd_res.kl_divergence(scd_res_dist_scd), 0.0
scd_res_dist_scd.kl_divergence(scd_res), 0.0

dist_res_scd = SCDSet.from_tree_distribution(dist_res)

dist_res_scd.kl_divergence(scd_res), 0.0
scd_res.kl_divergence(dist_res_scd), 0.0

dist_res_scd_dist = dist_res_scd.tree_distribution()

scd_res_dist.kl_divergence(dist_res_scd_dist), 0.0
dist_res_scd_dist.kl_divergence(scd_res_dist), 0.0

dist_res_scd.kl_divergence_scd_verbose(scd_res)
scd_res.kl_divergence_scd_verbose(dist_res_scd)

set(scd_res[scd_res.root_subsplit()].data.keys())==set(dist_res_scd[dist_res_scd.root_subsplit()].data.keys())

for child in scd_res[scd_res.root_subsplit()].data.keys():
    root = scd_res.root_subsplit()
    print(f"Child: {child} -- scd_res: {scd_res[root][child]:8.4g} -- dist_res_scd: {dist_res_scd[root][child]:8.4g}")

# SCD restriction properties

reload(classes)
from classes import *

root_clade = Clade("ABCDEF")
restriction = Clade("ABCD")

scd = SCDSet.random_with_sparsity(root_clade, sparsity=0.0)
dist = scd.tree_distribution()
dist_res = dist.restrict(restriction)

s_probs = scd.subsplit_probabilities()
restricted_subsplit = Subsplit("AB", "CD")

result = 0.0
for subsplit, prob in s_probs.items():
    if subsplit.restrict(restriction) == restricted_subsplit:
        result += prob

result
dist_res.feature_prob(restricted_subsplit)

# subsplit probability derivative

from importlib import reload
import classes
import random

reload(classes)
from classes import *


def scd_subsplit_derivative(prob_of: Subsplit, wrt: PCSS, scd: SCDSet, s2s: dict=None):
    root_subsplit = scd.root_subsplit()
    parent1 = wrt.parent
    child1 = wrt.child
    parent2 = prob_of.clade()
    if not prob_of.valid_ancestor(parent1):
        return 0.0
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    uncond_wrt = s2s.get(parent1, dict()).get(root_subsplit, 0.0) * scd[wrt]
    child_to_prob_of = s2s.get(prob_of, dict()).get(child1, 0.0)
    parent_to_prob_of = s2s.get(prob_of, dict()).get(parent1, 0.0)
    return uncond_wrt * (child_to_prob_of - parent_to_prob_of)


def scd_estimate_subsplit_derivative(prob_of: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001):
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    uncond = scd.subsplit_probabilities()
    uncond2 = scd2.subsplit_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

prob_of = random.choice(list(s2s.keys()))
wrt_parent = random.choice(list(s2s[prob_of].keys()))
wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of).keys()))
wrt = PCSS(wrt_parent, wrt_child)

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
uncond = scd.subsplit_probabilities()
uncond2 = scd2.subsplit_probabilities()
by_hand = (uncond2[prob_of] - uncond[prob_of]) / delta

est_deriv = scd_estimate_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd.tree_distribution()
tree_dist2 = scd2.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_num2 = tree_dist2.feature_prob(prob_of)
by_hand2 = (tree_num2 - tree_num) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

## All x all experiment

theo_res_simple = dict()
est_res_simple = dict()
for prob_of_for in scd.iter_subsplits():
    if prob_of_for not in theo_res_simple:
        theo_res_simple[prob_of_for] = dict()
    if prob_of_for not in est_res_simple:
        est_res_simple[prob_of_for] = dict()
    for wrt_for in scd.iter_pcss():
        theo_res_simple[prob_of_for][wrt_for] = scd_subsplit_derivative(prob_of=prob_of_for, wrt=wrt_for, scd=scd, s2s=s2s)
        est_res_simple[prob_of_for][wrt_for] = scd_estimate_subsplit_derivative(prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta)
        print(f"{prob_of_for} wrt {wrt_for}")
        print(f"Theo: {theo_res_simple[prob_of_for][wrt_for]:8.4g}")
        print(f"Est:  {est_res_simple[prob_of_for][wrt_for]:8.4g}")


for prob_of_for in theo_res_simple:
    for wrt_for in theo_res_simple[prob_of_for]:
        theo = theo_res_simple[prob_of_for][wrt_for]
        est = est_res_simple[prob_of_for][wrt_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 1e-4:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")


# subsplit to subsplit probability derivative

reload(classes)
from classes import *


def scd_subsplit_to_subsplit_cond_derivative(prob_of: Subsplit, cond_on: Subsplit, wrt: PCSS, scd: SCDSet, s2s: dict=None):
    # root_subsplit = scd.root_subsplit()
    parent1 = wrt.parent
    child1 = wrt.child
    # parent2 = prob_of.clade()
    if not prob_of.valid_ancestor(parent1):
        return 0.0
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    uncond_wrt = s2s.get(parent1, dict()).get(cond_on, 0.0) * scd[wrt]
    child_to_prob_of = s2s.get(prob_of, dict()).get(child1, 0.0)
    parent_to_prob_of = s2s.get(prob_of, dict()).get(parent1, 0.0)
    return uncond_wrt * (child_to_prob_of - parent_to_prob_of)


def scd_estimate_subsplit_to_subsplit_cond_derivative(prob_of: Subsplit, cond_on: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001):
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    s2s = scd.subsplit_to_subsplit_probabilities()
    s2s2 = scd2.subsplit_to_subsplit_probabilities()
    est_deriv = (s2s2[prob_of][cond_on] - s2s[prob_of][cond_on]) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

prob_of = random.choice(list(s2s.keys()))
wrt_parent = random.choice(list(s2s[prob_of].keys()))
wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of).keys()))
wrt = PCSS(wrt_parent, wrt_child)
cond_on = random.choice(list(s2s[wrt_parent].keys()))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
s2s = scd.subsplit_to_subsplit_probabilities()
s2s2 = scd2.subsplit_to_subsplit_probabilities()
by_hand = (s2s2[prob_of][cond_on] - s2s[prob_of][cond_on]) / delta

est_deriv = scd_estimate_subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=cond_on, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=cond_on, wrt=wrt, scd=scd, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}")

tree_dist = scd.tree_distribution()
tree_dist2 = scd2.tree_distribution()
tree_num = tree_dist.prob_all([prob_of, cond_on])
tree_num2 = tree_dist2.prob_all([prob_of, cond_on])
tree_den = tree_dist.feature_prob(cond_on)
tree_den2 = tree_dist2.feature_prob(cond_on)
tree_cond = tree_num/tree_den
tree_cond2 = tree_num2/tree_den2
by_hand2 = (tree_cond2 - tree_cond) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

# via experiment

reload(classes)
from classes import *


def scd_subsplit_via_subsplit_derivative(prob_of: Subsplit, via: Subsplit, wrt: PCSS, scd: SCDSet, s2s: dict=None):
    root_subsplit = scd.root_subsplit()
    parent1 = wrt.parent
    child1 = wrt.child
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    if parent1.valid_descendant(via):
        return scd_subsplit_to_subsplit_cond_derivative(prob_of=via, cond_on=root_subsplit, wrt=wrt, scd=scd, s2s=s2s) * s2s[prob_of][via]
    elif parent1.valid_ancestor(via) and parent1.valid_descendant(prob_of):
        return s2s[via][root_subsplit] * scd_subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=via, wrt=wrt, scd=scd, s2s=s2s)
    else:
        return 0.0


def scd_estimate_subsplit_via_subsplit_derivative(prob_of: Subsplit, via: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001):
    root_subsplit = scd.root_subsplit()
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    s2s = scd.subsplit_to_subsplit_probabilities()
    s2s2 = scd2.subsplit_to_subsplit_probabilities()
    est_deriv = (s2s2[prob_of][via]*s2s2[via][root_subsplit] - s2s[prob_of][via]*s2s[via][root_subsplit]) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
root_subsplit = scd.root_subsplit()
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

prob_of = random.choice(list(s2s.keys()))
wrt_parent = random.choice(list(s2s[prob_of].keys()))
wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of).keys()))
wrt = PCSS(wrt_parent, wrt_child)
via = random.choice(list(s2s[prob_of].keys()))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
s2s = scd.subsplit_to_subsplit_probabilities()
s2s2 = scd2.subsplit_to_subsplit_probabilities()
by_hand = (s2s2[prob_of][via]*s2s2[via][root_subsplit] - s2s[prob_of][via]*s2s[via][root_subsplit]) / delta

est_deriv = scd_estimate_subsplit_via_subsplit_derivative(prob_of=prob_of, via=via, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_subsplit_via_subsplit_derivative(prob_of=prob_of, via=via, wrt=wrt, scd=scd, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}")


# restricted unconditional derivative

reload(classes)
from classes import *


def scd_restricted_subsplit_derivative(restriction: Clade, prob_of: Subsplit, wrt: PCSS, scd: SCDSet, s2s: dict=None):
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    result = 0.0
    for subsplit in scd.iter_subsplits():
        restricted_subsplit = subsplit.restrict(restriction)
        if restricted_subsplit == prob_of:
            result += scd_subsplit_derivative(prob_of=subsplit, wrt=wrt, scd=scd, s2s=s2s)
    return result


def scd_estimate_restricted_subsplit_derivative(restriction: Clade, prob_of: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001):
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    scd_res = scd.restrict(restriction)
    scd2_res = scd2.restrict(restriction)
    uncond = scd_res.subsplit_probabilities()
    uncond2 = scd2_res.subsplit_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
restriction = Clade("ABCDE")
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

scd_res = scd.restrict(restriction)

prob_of = random.choice(list(scd_res.iter_subsplits()))
possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of]
prob_of_full = random.choice(possible)
wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
wrt = PCSS(wrt_parent, wrt_child)

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
scd2_res = scd2.restrict(restriction)

uncond = scd_res.subsplit_probabilities()
uncond2 = scd2_res.subsplit_probabilities()
by_hand = (uncond2[prob_of] - uncond[prob_of]) / delta

est_deriv = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_num2 = tree_dist2.feature_prob(prob_of)
by_hand2 = (tree_num2 - tree_num) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

## All x all experiment

theo_res = dict()
est_res = dict()
for prob_of_for in scd_res.iter_subsplits():
    if prob_of_for not in theo_res:
        theo_res[prob_of_for] = dict()
    if prob_of_for not in est_res:
        est_res[prob_of_for] = dict()
    for wrt_for in scd.iter_pcss():
        theo_res[prob_of_for][wrt_for] = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, s2s=s2s)
        est_res[prob_of_for][wrt_for] = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta)
        print(f"{prob_of_for} wrt {wrt_for}")
        print(f"Theo: {theo_res[prob_of_for][wrt_for]:8.4g}")
        print(f"Est:  {est_res[prob_of_for][wrt_for]:8.4g}")


# AE:BC wrt ABCE:DF, D:F
# Theo: -0.005769
# Est:         0

for prob_of_for in theo_res:
    for wrt_for in theo_res[prob_of_for]:
        theo = theo_res[prob_of_for][wrt_for]
        est = est_res[prob_of_for][wrt_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 1e-4:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")

# restricted PCSS derivative

reload(classes)
from classes import *


def scd_restricted_pcss_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, s2s: dict=None):
    root_subsplit = scd.root_subsplit()
    parent = prob_of.parent
    child = prob_of.child
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    result = 0.0
    for destination in s2s:
        dest_res = destination.restrict(restriction)
        if dest_res != child:
            continue
        for origin in s2s[destination]:
            orig_res = origin.restrict(restriction)
            if orig_res != parent or (orig_res.is_trivial() and origin != root_subsplit):
                continue
            result += scd_subsplit_via_subsplit_derivative(prob_of=destination, via=origin, wrt=wrt, scd=scd, s2s=s2s)
    return result


def scd_estimate_restricted_pcss_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, delta: float=0.0001):
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    scd_res = scd.restrict(restriction)
    scd2_res = scd2.restrict(restriction)
    uncond = scd_res.pcss_probabilities()
    uncond2 = scd2_res.pcss_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
restriction = Clade("ABCD")
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

scd_res = scd.restrict(restriction)

prob_of = random.choice(list(scd_res.iter_pcss()))
possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of.parent]
prob_of_full = random.choice(possible)
wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
wrt = PCSS(wrt_parent, wrt_child)

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
scd2_res = scd2.restrict(restriction)

uncond = scd_res.pcss_probabilities()
uncond2 = scd2_res.pcss_probabilities()
by_hand = (uncond2[prob_of] - uncond[prob_of]) / delta

est_deriv = scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_num2 = tree_dist2.feature_prob(prob_of)
by_hand2 = (tree_num2 - tree_num) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

# restricted conditional derivative


reload(classes)
from classes import *


def scd_restricted_conditional_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, s2s: dict=None, restricted_scd: SCDSet=None):
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
    restricted_pcss_probs = restricted_scd.pcss_probabilities()
    # Quotient rule d(top/bot) = (bot*dtop-top*dbot) / bot**2
    top = restricted_pcss_probs[prob_of]
    print(f"top: {top}")
    bot = restricted_subsplit_probs[prob_of.parent]
    print(f"bot: {bot}")
    dtop = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, s2s=s2s)
    print(f"dtop: {dtop}")
    dbot = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, s2s=s2s)
    print(f"dbot: {dbot}")
    return (bot*dtop - top*dbot) / bot**2


def scd_estimate_restricted_conditional_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, delta: float=0.0001):
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    scd_res = scd.restrict(restriction)
    scd2_res = scd2.restrict(restriction)
    est_deriv = (scd2_res[prob_of] - scd_res[prob_of]) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
restriction = Clade("ABCDE")
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

scd_res = scd.restrict(restriction)

# prob_of = random.choice(list(scd_res.iter_pcss()))
# possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of.parent]
# prob_of_full = random.choice(possible)
# wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
# wrt = PCSS(wrt_parent, wrt_child)

# prob_of = PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C"))
# prob_of=PCSS(Subsplit("ACE", "BD"), Subsplit("AC", "E"))
# wrt = PCSS(Subsplit("ABCDEF"), Subsplit("ACEF", "BD"))
prob_of = PCSS(Subsplit("ADE", "B"), Subsplit("AD", "E"))
wrt = PCSS(Subsplit(Clade("ADE"), Clade("BF")), Subsplit(Clade("A"), Clade("DE")))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
scd2_res = scd2.restrict(restriction)

by_hand = (scd2_res[prob_of] - scd_res[prob_of]) / delta
by_hand

est_deriv = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_den = tree_dist.feature_prob(prob_of.parent)
tree_num2 = tree_dist2.feature_prob(prob_of)
tree_den2 = tree_dist2.feature_prob(prob_of.parent)
by_hand2 = (tree_num2/tree_den2 - tree_num/tree_den) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, s2s=s2s)
scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, s2s=s2s)
scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, delta=delta)

# restricted KL derivative


reload(classes)
from classes import *


def scd_restricted_kl_derivative(wrt: PCSS, scd: SCDSet, other: SCDSet, s2s: dict=None, restricted_scd: SCDSet=None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    # restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
    # restricted_pcss_probs = restricted_scd.pcss_probabilities()
    other_pcss_probs = other.pcss_probabilities()
    result = 0.0
    for other_pcss in other.iter_pcss():
        other_pcss_prob = other_pcss_probs[other_pcss]
        # print(f"other_pcss_prob={other_pcss_prob}")
        restricted_cond = restricted_scd[other_pcss]
        # print(f"restricted_cond={restricted_cond}")
        print(f"restriction={restriction}, prob_of={other_pcss}, wrt={wrt}")
        restricted_cond_deriv = scd_restricted_conditional_derivative(restriction=restriction, prob_of=other_pcss, wrt=wrt, scd=scd, s2s=s2s, restricted_scd=restricted_scd)
        print(f"restricted_cond_deriv={restricted_cond_deriv}")
        intermediate_result = -other_pcss_prob * restricted_cond_deriv / restricted_cond
        print(f"intermediate_result={intermediate_result}")
        result += intermediate_result

    return result


def scd_estimate_restricted_kl_derivative(wrt: PCSS, scd: SCDSet, other: SCDSet, delta: float=0.0001):
    parent = wrt.parent
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(parent)
    scd_res = scd.restrict(restriction)
    scd2_res = scd2.restrict(restriction)
    est_deriv = (other.kl_divergence(scd2_res) - other.kl_divergence(scd_res)) / delta
    return est_deriv


scd = SCDSet.random("ABCDEF")
restriction = Clade("ABCDE")
scd_small = SCDSet.random(restriction)
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

scd_res = scd.restrict(restriction)

wrt = random.choice(list(scd.iter_pcss()))
# possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of.parent]
# prob_of_full = random.choice(possible)
# wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
# wrt = PCSS(wrt_parent, wrt_child)

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
scd2_res = scd2.restrict(restriction)

by_hand = (scd_small.kl_divergence(scd2_res) - scd_small.kl_divergence(scd_res)) / delta
by_hand

est_deriv = scd_estimate_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, delta=delta)
theo_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, s2s=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
small_tree_dist = scd_small.tree_distribution()
by_hand2 = (small_tree_dist.kl_divergence(tree_dist2) - small_tree_dist.kl_divergence(tree_dist)) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

test_prob_of = PCSS(Subsplit("ADE", "B"), Subsplit("AD", "E"))
test_wrt=PCSS(Subsplit(Clade("ADE"), Clade("BF")), Subsplit(Clade("A"), Clade("DE")))
scd_restricted_conditional_derivative(restriction=restriction, prob_of=test_prob_of, wrt=test_wrt, scd=scd, s2s=s2s)
scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=test_prob_of, wrt=test_wrt, scd=scd, delta=delta)





