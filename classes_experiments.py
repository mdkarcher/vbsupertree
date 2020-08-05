from importlib import reload
from collections import OrderedDict
from more_itertools import powerset

from small_trees import *

import classes
import derivatives

reload(classes)
from classes import *

reload(derivatives)
from derivatives import *

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

# Testing new SCD backend

reload(classes)
from classes import *

# pcss_conditional_probs = {
#     PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
#     PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
#     PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
#     PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
#     PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
#     PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
#     PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
#     PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
# }
# scd = SCDSet(pcss_conditional_probs)
# scd.normalize()

scd = SCDSet.random("ABCDE")
tree_dist = scd.tree_distribution()
parent_probs = scd.subsplit_probabilities()
parent, prob = random.choice(list(parent_probs.items()))
print(f"{tree_dist.feature_prob(parent)}\n{prob}")
for parent, prob in parent_probs.items():
    tree_prob = tree_dist.feature_prob(parent)
    if abs(tree_prob - prob) > 1e-15:
        print(f"{parent}\n{tree_prob}\n{prob}")

pcss_probs = scd.pcss_probabilities()
pcss, prob = random.choice(list(pcss_probs.items()))
print(f"{tree_dist.prob_all([pcss.parent, pcss.child])}\n{prob}")
for pcss, prob in pcss_probs.items():
    tree_prob = tree_dist.prob_all([pcss.parent, pcss.child])
    if abs(tree_prob - prob) > 1e-15:
        print(f"{pcss}\n{tree_prob}\n{prob}")

subsplit_probs, pcss_probs = scd.probabilities()
for parent, prob in parent_probs.items():
    tree_prob = tree_dist.feature_prob(parent)
    if abs(tree_prob - prob) > 1e-15:
        print(f"{parent}\n{tree_prob}\n{prob}")
for pcss, prob in pcss_probs.items():
    tree_prob = tree_dist.prob_all([pcss.parent, pcss.child])
    if abs(tree_prob - prob) > 1e-15:
        print(f"{pcss}\n{tree_prob}\n{prob}")

# New SCD transit probabilities

reload(classes)
from classes import *

root_clade = Clade("ABCDE")
root_subsplit = Subsplit(root_clade)
scd = SCDSet.random(root_clade)
tree_dist = scd.tree_distribution()

t_probs = scd.transit_probabilities()
s_probs = scd.subsplit_probabilities()

parent = Subsplit(Clade("ABD"), Clade("C"))
s_probs[parent]
# t_probs[parent]
t_probs[parent][root_subsplit]

t_probs[parent][parent], 1.0

child = Subsplit(Clade("A"), Clade("BD"))
den = tree_dist.feature_prob(parent)
num = tree_dist.prob_all([parent, child])
t_probs[child][parent], num / den

## All x all experiment


def estimate_transit_probability(tree_dist: TreeDistribution, ancestor, descendant: Subsplit):
    descendant_clade = descendant.clade()
    if isinstance(ancestor, Subsplit):
        if ancestor == descendant:
            return 1.0
        if not (descendant_clade.issubset(ancestor.clade1) or descendant_clade.issubset(ancestor.clade2)):
            return 0.0
    if isinstance(ancestor, SubsplitClade):
        if not descendant_clade.issubset(ancestor.clade):
            return 0.0
    den = tree_dist.feature_prob(ancestor)
    if den == 0.0:
        return 0.0
    num = tree_dist.prob_all([ancestor, descendant])
    return num / den


estimate_transit_probability(tree_dist, parent, child)
t_probs[child][parent]

result = dict()
for ancestor in scd.iter_parents():
    print(f"Ancestor = {ancestor}")
    if ancestor not in result:
        result[ancestor] = dict()
    for descendant in scd.iter_subsplits():
        print(f"\tDescendant = {descendant}")
        est = estimate_transit_probability(tree_dist, ancestor, descendant)
        calc = t_probs.get(descendant, dict()).get(ancestor, 0.0)
        result[ancestor][descendant] = est, calc
        if abs(est - calc) > 1e-9:
            # print(f"{ancestor} to {descendant} : Error!")
            print(f"{ancestor} to {descendant} :\nEst={est}\nCal={calc}")
        else:
            pass
            # print(f"{ancestor} to {descendant} : Correct")

for ancestor in result:
    for descendant in result[ancestor]:
        est, calc = result[ancestor][descendant]
        if abs(est - calc) > 1e-9:
            # print(f"{ancestor} to {descendant} : Error!")
            print(f"{ancestor} to {descendant} :\nEst={est}\nCal={calc}")
        else:
            # print(f"{ancestor} to {descendant} : Correct")
            pass

ancestor = SubsplitClade(Subsplit("ABCDE", ""), Clade("ABCDE"))
descendant = Subsplit("ABE", "CD")
estimate_transit_probability(tree_dist, ancestor, descendant)
t_probs[descendant][ancestor]

result = dict()
for ancestor in scd.iter_subsplits():
    if ancestor not in result:
        result[ancestor] = dict()
    for descendant in scd.iter_subsplits():
        est = estimate_transit_probability(tree_dist, ancestor, descendant)
        calc = t_probs.get(descendant, dict()).get(ancestor, 0.0)
        result[ancestor][descendant] = est, calc
        if abs(est - calc) > 1e-9:
            # print(f"{ancestor} to {descendant} : Error!")
            print(f"{ancestor} to {descendant} :\nEst={est}\nCal={calc}")
        else:
            # print(f"{ancestor} to {descendant} : Correct")
            pass

# New PCSSSupport tests

reload(classes)
from classes import *

taxa = Clade("ABCDE")
tree = MyTree.random(taxa)
pcsss = PCSSSupport.from_tree(tree)
pcsss.to_set()
pcsss.get_taxon_set()

# new SCDSet restrict

reload(classes)
from classes import *

taxa = Clade("ABCDEF")
restriction = Clade("ABCDE")

scd = SCDSet.random(taxa)
t_probs = scd.transit_probabilities()

root_subsplit_clade = scd.root_subsplit_clade()
root_subsplit = scd.root_subsplit()
t_probs[root_subsplit]


# Testing PCSSSupport.restrict

reload(classes)
from classes import *


X = Clade("ABCDEFG")
restriction = Clade("ABCDE")

flat_scd = SCDSet.random(X, concentration=10)
trees = set()
while len(trees) < 25:
    trees.add(flat_scd.random_tree())

true_tree_dist = TreeDistribution.random(trees)
true_scd = SCDSet.from_tree_distribution(true_tree_dist)
true_support = true_scd.support()

res_support = true_support.restrict(restriction)

# Testing PCSSSupport pruning

reload(classes)
from classes import *


X1 = Clade("ABCDEF")
X2 = Clade("ABCDEG")
scd1 = SCDSet.random_with_sparsity(X1, sparsity=0.3)
scd2 = SCDSet.random_with_sparsity(X2, sparsity=0.3)
support1 = scd1.support()
support2 = scd2.support()
mutual_support = support1.mutualize(support2)
mutual_support.is_complete()
len(mutual_support)

pruned_support = mutual_support.prune(verbose=True)
len(pruned_support)
pruned_support.is_complete()


# Testing SCDSet pruning

reload(classes)
from classes import *

X = Clade("ABCDEF")
scd = SCDSet.random_with_sparsity(X, 0.9)
len(scd)
tree_dist = scd.tree_distribution()
len(tree_dist)

parent = Subsplit(Clade("ABDE"), Clade("CF"))
child = Subsplit(Clade("AE"), Clade("BD"))
pcss = PCSS(parent, child)
scd[pcss]

scd.remove(pcss)
scd
pruned_scd = scd.prune(verbose=True)
pruned_scd.normalize()
pruned_scd


