from importlib import reload
from collections import OrderedDict
from more_itertools import powerset

from small_trees import *

import classes
reload(classes)

from classes import *

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

# PCSS experiments

from importlib import reload
import classes

reload(classes)
from classes import *

tree = MyTree.random("ABCDE")
print(tree)
list(tree.traverse_subsplits())
list(tree.traverse_pcsss())

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
# result = PCSSSupport_old()
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

scd = SCDSet_old.random("ABCD")
scd
len(scd)
scd.check()

tree = scd.random_tree()
print(tree)

big_scd = SCDSet_old.random("ABCDEFG")
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
scd = SCDSet_old(pcss_conditional_probs)
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

scd = SCDSet_old.random("ABCDE")
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

scd = SCDSet_old.random("ABCDE")
tree_dist = scd.tree_distribution()
parent_probs = scd.subsplit_probabilities()

parent, prob = random.choice(list(parent_probs.items()))

tree_dist.feature_prob(parent)
prob

# pcss_probabilities experiment

import random

reload(classes)
from classes import *

scd = SCDSet_old.random("ABCDE")
tree_dist = scd.tree_distribution()
pcss_probs = scd.pcss_probabilities()

pcss, prob = random.choice(list(pcss_probs.items()))

tree_dist.prob_all([pcss.parent, pcss.child])
prob

# SCDSet_old kl implementations

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
scd1 = SCDSet_old(pcss_conditional_probs1)
scd1.normalize()
dist1 = scd1.tree_distribution()
scd2 = SCDSet_old(pcss_conditional_probs2)
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

scd = SCDSet_old.random_with_sparsity(root_clade, sparsity=0.0)
dist = scd.tree_distribution()
dist_res = dist.restrict(restriction)

scd_res = scd.restrict(restriction, verbose=True)
scd_res_dist = scd_res.tree_distribution()

scd_res_dist_scd = SCDSet_old.from_tree_distribution(scd_res_dist)

scd_res.kl_divergence(scd_res_dist_scd), 0.0
scd_res_dist_scd.kl_divergence(scd_res), 0.0

dist_res_scd = SCDSet_old.from_tree_distribution(dist_res)

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

scd = SCDSet_old.random_with_sparsity(root_clade, sparsity=0.0)
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
