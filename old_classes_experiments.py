from importlib import reload
from collections import OrderedDict
from more_itertools import powerset

from small_trees import *

import classes
reload(classes)

from classes import *

ccd = CCDSet({Subsplit("AB", "CD"): 0.4, Subsplit("ABC", "D"): 0.6,
              Subsplit("AB", "C"): 2/3, Subsplit("A", "BC"): 1/3,
              Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
              Subsplit("C", "D"): 1.0})
dist = ccd.tree_distribution()
print(dist)

res_dist = dist.restrict("ABC")
print(res_dist)

res_ccd = ccd.restrict("ABC")
res_ccd.check()
res_ccd.is_complete()
res_ccd.check_distributions()
res_dist2 = res_ccd.tree_distribution()
print(res_dist2)

clade = Clade("ABC")
size = len(clade)
for k in range(1, size // 2 + 1):
    for left in combinations(clade, k):
        left_clade = Clade(left)
        print(left_clade)
        print(clade - left_clade)
        print(Subsplit(left_clade, clade - left_clade))


list(Subsplit.compatible_subsplits(clade))

ProbabilityDistribution.random("ABCDE", 0.1)

pr = ProbabilityDistribution({"A": 0.4, "B": 0.4})
pr2 = pr.normalize()
print(pr2)


pr = ProbabilityDistribution.random("ABCDE", concentration=0.1)
print(pr)
pr2 = pr.trim_small(0.1).normalize()
print(pr2)

pr = ProbabilityDistribution.random("ABCDE", concentration=0.1)
pr2 = ProbabilityDistribution(pr)
print(pr)
print(pr2)
pr == pr2

# CCD Restriction Experiment
big_clade = "ABCDEF"
small_clade = "ABCDE"

ccd = CCDSet.random(big_clade, 1.0, 0.03)
print(ccd)

dist = ccd.tree_distribution()
print(dist)

res_dist = dist.restrict(small_clade)
print(res_dist)

res_ccd = ccd.restrict(small_clade)
print(res_ccd)

res_dist2 = res_ccd.tree_distribution()
print(res_dist2)

for tree in res_dist:
    print(res_dist[tree] - res_dist2[tree])
res_dist.kl_divergence(res_dist2)

# Restricted CCD Existence Experiment

big_clade = "ABCDEFG"
small_clade = "ABCDE"

ccd = CCDSet.random(big_clade, 1.0, 0.02)
print(ccd)

ccd.check()
for bad_clade in ccd.bad_distributions():
    # print(str(bad_clade) + ": " + str(sum(ccd[bad_clade].values())))
    print(str(bad_clade) + ": " + str(round(sum(ccd[bad_clade].values()), 9)))


dist = ccd.tree_distribution()
print(dist)

# ccd2 = dist.to_ccd()
# dist2 = ccd.tree_distribution()
# for tree in dist:
#     print(dist[tree] - dist2[tree])
# dist.kl_divergence(dist2)

res_dist = dist.restrict(small_clade)
print(res_dist)

res_ccd = res_dist.to_ccd()
print(res_ccd)

res_dist2 = res_ccd.tree_distribution()
print(res_dist2)

for tree in res_dist:
    print(round(res_dist[tree] - res_dist2[tree], 9))
print(round(res_dist.kl_divergence(res_dist2), 9))

# for tree in res_dist:
#     print(res_dist[tree] - res_dist2[tree])
# res_dist.kl_divergence(res_dist2)

alt_res_ccd = ccd.restrict(small_clade)
for bad_clade in alt_res_ccd.bad_distributions():
    print(alt_res_ccd[bad_clade])
    print(sum(alt_res_ccd[bad_clade].values()))

alt_res_dist = alt_res_ccd.tree_distribution()

print("KL: " + str(round(res_dist.kl_divergence(alt_res_dist), 9)))

# Quick-repeat restricted CCD existence experiment


def restriction_test_function(big_clade, small_clade, concentration=1.0, cutoff=0.05):
    ccd = CCDSet.random(big_clade, concentration, cutoff)
    # print(ccd.check())
    # for bad_clade in ccd.bad_distributions():
    #     print(str(bad_clade) + ": " + str(round(sum(ccd[bad_clade].values()), 9)))
    print(all(round(sum(ccd[bad_clade].values()), 9)==1.0 for bad_clade in ccd.bad_distributions()))
    dist = ccd.tree_distribution()
    res_dist = dist.restrict(small_clade)
    res_ccd = res_dist.to_ccd()
    res_dist2 = res_ccd.tree_distribution()
    # for tree in res_dist:
    #     print(round(abs(res_dist[tree] - res_dist2[tree]), 9))
    print(all(round(abs(res_dist[tree] - res_dist2[tree]), 9) == 0.0 for tree in res_dist))
    print("KL: " + str(round(res_dist.kl_divergence(res_dist2), 9)))
    # print(ccd.check())
    alt_res_ccd = ccd.restrict(small_clade)
    for bad_clade in alt_res_ccd.bad_distributions():
        print(alt_res_ccd[bad_clade])
        print(sum(alt_res_ccd[bad_clade].values()))

    alt_res_dist = alt_res_ccd.tree_distribution()

    for tree in res_dist:
        print(round(abs(res_dist[tree] - alt_res_dist[tree]), 9))
    print("KL: " + str(round(res_dist.kl_divergence(alt_res_dist), 9)))
    return ccd, dist, res_dist, res_ccd, res_dist2, alt_res_ccd, alt_res_dist


big_clade = "ABCDEFG"
small_clade = "ABCDEF"
ccd, dist, res_dist, res_ccd, res_dist2, alt_res_ccd, alt_res_dist = restriction_test_function(big_clade, small_clade, cutoff=0.02)

print(ccd)

for bad_clade in ccd.bad_distributions():
    print(str(bad_clade) + ": " + str(sum(ccd[bad_clade].values())))

res_dist.kl_divergence(alt_res_dist)
for bad_clade in alt_res_ccd.bad_distributions():
    print(alt_res_ccd[bad_clade])
    print(sum(alt_res_ccd[bad_clade].values()))
print(ccd)
print(res_ccd)

# Direct CCD KL comparison

ccd = CCDSet({Subsplit("AB", "CD"): 0.4, Subsplit("ABC", "D"): 0.6,
              Subsplit("AB", "C"): 2/3, Subsplit("A", "BC"): 1/3,
              Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
              Subsplit("C", "D"): 1.0})
dist = ccd.tree_distribution()
print(dist)

ccd2 = CCDSet({Subsplit("AB", "CD"): 0.5, Subsplit("ABC", "D"): 0.5,
              Subsplit("AB", "C"): 0.5, Subsplit("A", "BC"): 0.5,
              Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
              Subsplit("C", "D"): 1.0})
dist2 = ccd2.tree_distribution()
print(dist2)

dist.kl_divergence(dist2)
ccd.kl_divergence(ccd2)

# %timeit dist.kl_divergence(dist2)
# %timeit ccd.kl_divergence(ccd2)

dist.kl_divergence(ccd2)
ccd.kl_divergence(dist2)

# %timeit dist.kl_divergence(ccd2)
# %timeit ccd.kl_divergence(dist2)

# CCD -> restricted CCD vs CCD -> tree dist -> restricted tree dist comparison

big_clade = Clade("ABCDEFG")
small_clade = Clade("ACDEG")

big_ccd = CCDSet.random(big_clade)
small_ccd = CCDSet.random(small_clade)

big_dist = big_ccd.tree_distribution()
small_dist = small_ccd.tree_distribution()

small_dist_approx1 = big_dist.restrict(small_clade)

small_ccd.kl_divergence(small_dist_approx1)
small_dist.kl_divergence(small_dist_approx1)

small_ccd2 = big_ccd.restrict(small_clade)

small_ccd.kl_divergence(small_ccd2)
small_dist.kl_divergence(small_ccd2)

small_dist2 = small_ccd2.tree_distribution()
small_dist.kl_divergence(small_dist2)

small_ccd3 = small_dist2.to_ccd()
small_ccd3 == small_ccd2
small_ccd3.kl_divergence(small_ccd2)
print(small_ccd2)
print(small_ccd3)

# Model is correct case

big_clade = Clade("ABCDEFG")
small_clade = Clade("ABCDEF")

big_ccd = CCDSet.random(big_clade, cutoff=0.00)
small_ccd = big_ccd.restrict(small_clade)

big_dist = big_ccd.tree_distribution()
small_dist = small_ccd.tree_distribution()

small_dist_approx1 = big_dist.restrict(small_clade)

small_ccd.kl_divergence(small_dist_approx1)
small_dist.kl_divergence(small_dist_approx1)

small_ccd2 = big_ccd.restrict(small_clade)

small_ccd.kl_divergence(small_ccd2)
small_dist.kl_divergence(small_ccd2)

small_ccd_approx1 = small_dist_approx1.to_ccd()
small_ccd.kl_divergence(small_ccd_approx1)

##

big_clade = Clade("ABCDEFG")
small_clade = Clade("ACDEG")

big_ccd = CCDSet.random(big_clade, cutoff=0.02)
small_ccd = big_ccd.restrict(small_clade)
small_ccd2 = big_ccd.restrict(small_clade)

small_ccd.kl_divergence(small_ccd2)
small_ccd2.kl_divergence(small_ccd)

small_dist = small_ccd.tree_distribution()
small_dist2 = small_ccd2.tree_distribution()

small_dist.kl_divergence(small_dist2)
small_dist2.kl_divergence(small_dist)

small_dist.kl_divergence(small_ccd2)
small_ccd.kl_divergence(small_dist2)

# KL testing

clade = Clade("ABCDEFG")
ccd1 = CCDSet.random(clade)
ccd2 = CCDSet.random(clade)
dist1 = ccd1.tree_distribution()
dist2 = ccd2.tree_distribution()
ccd1.kl_divergence(ccd2)
ccd1.kl_divergence(dist2)
dist1.kl_divergence(ccd2)
dist1.kl_divergence(dist2)

# %timeit ccd1.kl_divergence(ccd2)
# %timeit ccd1.kl_divergence(dist2)
# %timeit dist1.kl_divergence(ccd2)
# %timeit dist1.kl_divergence(dist2)

# Subsplit support mutualization experiment

X = "ABCD"
all_trees = MyTree.generate_rooted(X)

big_tree = next(all_trees)
print(big_tree)

leaf_set1, leaf_set2 = "ABC", "BCD"
# tree1 = restrict_tree(big_tree, leaf_set1)
# tree2 = restrict_tree(big_tree, leaf_set2)
tree1 = big_tree.restrict(leaf_set1)
tree2 = big_tree.restrict(leaf_set2)
print(tree1)
print(tree2)

# manifest1 = get_subsplit_manifest(tree1)
# manifest2 = get_subsplit_manifest(tree2)

# manifest12 = generate_mutual_manifest(manifest1, manifest2, verbose=True)
# print_manifest(manifest12)

ss1 = SubsplitSupport.from_tree(tree1)
ss2 = SubsplitSupport.from_tree(tree2)
print(ss1)
print(ss2)
ss = ss1.mutualize(ss2)
print(ss)
big_ss = SubsplitSupport.from_tree(big_tree)
print(big_ss)
big_ss.to_set().issubset(ss.to_set())

# Continued

X = "ABCDE"
ambig = 1
n = 2
restrictions = full_cover_restrictions_n(X, ambig, n)
restriction = next(restrictions)

result = visualize_restricted_rooted_trees_n(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i1 = [5, 25, 67]
pjts1 = [result[result_keys[i]] for i in i1]
big_tree1 = [pjt.treelist[0] for pjt in pjts1]
projection1 = [pjt.projections[0] for pjt in pjts1]


i2 = [19, 44, 72]
pjts2 = [result[result_keys[i]] for i in i2]
big_tree2 = [pjt.treelist[0] for pjt in pjts2]
projection2 = [pjt.projections[1] for pjt in pjts2]

manifests1 = [get_subsplit_manifest(proj) for proj in projection1]
manifests2 = [get_subsplit_manifest(proj) for proj in projection2]

merged_manifest1 = merge_manifests(manifests1)
merged_manifest2 = merge_manifests(manifests2)

manifest12 = generate_mutual_manifest(merged_manifest1, merged_manifest2, verbose=True)
print_manifest(manifest12, show_degenerate=False)

ss1 = SubsplitSupport.from_trees(projection1)
ss2 = SubsplitSupport.from_trees(projection2)
ss = ss1.mutualize(ss2)
print(ss)

# PCSSSupport_old test

X = "ABCD"
all_trees = generate_rooted(X)

big_tree = next(all_trees)
print(big_tree)

leaf_set1, leaf_set2 = "ABC", "BCD"
tree1 = restrict_tree(big_tree, leaf_set1)
tree2 = restrict_tree(big_tree, leaf_set2)
print(tree1)
print(tree2)

manifest1 = get_subsplit_manifest(tree1)
manifest2 = get_subsplit_manifest(tree2)
manifest12 = generate_mutual_manifest(manifest1, manifest2, verbose=True)
print_manifest(manifest12)

ss1 = SubsplitSupport.from_tree(tree1)
ss2 = SubsplitSupport.from_tree(tree2)
print(ss1)
print(ss2)
ss = ss1.mutualize(ss2)
print(ss)

ps = PCSSSupport_old.from_tree(big_tree)
print(ps)
ps.is_complete()
tree = ps.random_tree()
print(tree)

ps1 = PCSSSupport_old.from_tree(tree1)
print(ps1)
ps1.is_complete()

ps2 = PCSSSupport_old.from_tree(tree2)
print(ps2)
ps1.is_complete()

pshat = ps1.mutualize(ps2)
print(pshat)
pshat.is_complete()


trees = ss.all_trees()
ps = PCSSSupport_old.from_trees(trees)
print(ps)
trees2 = ps.all_trees()
for tree in trees: print(tree)
for tree in trees2: print(tree)

# Disambiguation experiment

X = "ABCD"
ambig = 1
restrictions = full_cover_restrictions(X, ambig)
restriction = next(restrictions)
print(restriction)

result = visualize_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i = 0
print(result[result_keys[i]].proj1)
print(result[result_keys[i]].proj2)
for tree in result[result_keys[i]].treelist: print(tree)

example_tree = MyTree(result[result_keys[i]].treelist[0])
print(example_tree)
example_tree1 = MyTree(result[result_keys[i]].proj1)
print(example_tree1)
example_tree2 = MyTree(result[result_keys[i]].proj2)
print(example_tree2)

ps = PCSSSupport_old.from_tree(example_tree)
print(ps)
ps1 = PCSSSupport_old.from_tree(example_tree1)
print(ps1)
ps2 = PCSSSupport_old.from_tree(example_tree2)
print(ps2)
pshat = ps1.mutualize(ps2)
print(pshat)

# Too big too small experiment


def enumerate_restricted_rooted_trees(taxa, restriction):
    result = OrderedDict()
    for tree in MyTree.generate_rooted(taxa):
        signature = tuple(tree.restrict(clade) for clade in restriction)
        if signature not in result:
            result[signature] = ProjectionTrees_n(signature, [])
        result[signature].treelist.append(tree)
    return result


X = "ABCDEFG"
ambig = 3
n = 2
restrictions = full_cover_restrictions_n(X, ambig, n)
restriction = next(restrictions)

result = enumerate_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i = 13
for tree in result[result_keys[i]].projections: print(tree)
for tree in result[result_keys[i]].treelist: print(tree)
trees = set(result[result_keys[i]].treelist)
proj1, proj2 = result[result_keys[i]].projections
ss1 = SubsplitSupport.from_tree(proj1)
ss2 = SubsplitSupport.from_tree(proj2)
sshat = ss1.mutualize(ss2)
sshat_trees = set(sshat.all_trees())
trees == sshat_trees
trees.issubset(sshat_trees)
ps1 = PCSSSupport_old.from_tree(proj1)
ps2 = PCSSSupport_old.from_tree(proj2)
pshat = ps1.mutualize(ps2)
pshat_trees = set(pshat.all_trees())
trees == pshat_trees
trees.issubset(pshat_trees)

pshat.all_strictly_complete_supports(True)  # Bugs in here

print("Subsplit Support Test")
for i, key in enumerate(result_keys):
    trees = set(result[result_keys[i]].treelist)
    proj1, proj2 = result[result_keys[i]].projections
    ss1 = SubsplitSupport.from_tree(proj1)
    ss2 = SubsplitSupport.from_tree(proj2)
    sshat = ss1.mutualize(ss2)
    sshat_trees = set(sshat.all_trees())
    print(f"Signature {i}: {len(trees)} trees, {len(sshat_trees)} reconstructed trees; same? {trees == sshat_trees}, subset? {trees.issubset(sshat_trees)}")

print("PCSS Support Test")
for i, key in enumerate(result_keys):
    trees = set(result[result_keys[i]].treelist)
    proj1, proj2 = result[result_keys[i]].projections
    ps1 = PCSSSupport_old.from_tree(proj1)
    ps2 = PCSSSupport_old.from_tree(proj2)
    pshat = ps1.mutualize(ps2)
    pshat_trees = set(pshat.all_trees())
    print(f"Signature {i}: {len(trees)} trees, {len(pshat_trees)} reconstructed trees; same? {trees == pshat_trees}, subset? {trees.issubset(pshat_trees)}")

print("Comparison Test")
for i, key in enumerate(result_keys):
    trees = set(result[result_keys[i]].treelist)
    proj1, proj2 = result[result_keys[i]].projections
    ss1 = SubsplitSupport.from_tree(proj1)
    ss2 = SubsplitSupport.from_tree(proj2)
    sshat = ss1.mutualize(ss2)
    sshat_trees = set(sshat.all_trees())
    ps1 = PCSSSupport_old.from_tree(proj1)
    ps2 = PCSSSupport_old.from_tree(proj2)
    pshat = ps1.mutualize(ps2)
    pshat_trees = set(pshat.all_trees())
    print(f"Signature {i:>3}: {len(trees):>2} trees, {len(sshat_trees):>2} SS-reconstructed trees, {len(pshat_trees):>2} PC-reconstructed trees; {trees == sshat_trees:>5} {trees == pshat_trees:>5} {trees.issubset(sshat_trees):>5} {trees.issubset(pshat_trees):>5}")
    # print(f"   Matches SS?  {trees == sshat_trees},  matches PC?  {trees == pshat_trees},  subset SS?  {trees.issubset(sshat_trees)},  subset PC?  {trees.issubset(pshat_trees)}")

# Experiment: take two random joint trees, restrict, reconstruct,
# check if originals are in reconstruction, compare reconstruction sizes.

X = "ABCDEFG"
ambig = 1
n = 2
restrictions = full_cover_restrictions_n(X, ambig, n)
restriction = next(restrictions)
rest1, rest2 = restriction
k = 3

trees = [MyTree.random(X) for _ in range(k)]
proj1s = [tree.restrict(rest1) for tree in trees]
proj2s = [tree.restrict(rest2) for tree in trees]
ss1 = SubsplitSupport.from_trees(proj1s)
ss2 = SubsplitSupport.from_trees(proj2s)
sshat = ss1.mutualize(ss2)
sshat_trees = set(sshat.all_trees())
ps1 = PCSSSupport_old.from_trees(proj1s)
ps2 = PCSSSupport_old.from_trees(proj2s)
pshat = ps1.mutualize(ps2)
pshat_trees = set(pshat.all_trees())

pshat.is_complete(verbose=True)
pshat.to_string_dict()

SubsplitSupport.from_trees(trees).to_set().issubset(sshat.to_set())
PCSSSupport_old.from_trees(trees).to_set().issubset(pshat.to_set())
set(trees).issubset(sshat_trees)
set(trees).issubset(pshat_trees)
len(sshat_trees)
len(pshat_trees)

# Newick test

tree1 = ete3.Tree("((((A,E),D),C),B);")
print(tree1)
tree2 = ete3.Tree("((((A,D),E),B),C);")
print(tree2)
ss = SubsplitSupport.from_trees([tree1, tree2])
print(ss)
len(ss.all_trees())  # 4
ps = PCSSSupport_old.from_trees([tree1, tree2])
print(ps)
len(ps.all_trees())  # 2

# Newick test 2

tree = MyTree("((((((A,E),D),C),B),F),G);")
# tree1 = MyTree("(((((A,E),D),C),B),F);")
# print(tree1)
# tree2 = MyTree("(((((A,D),E),B),C),G);")
# print(tree2)
proj1 = tree.restrict("ABCDEF")
proj2 = tree.restrict("ABCDEG")

ss1 = SubsplitSupport.from_tree(proj1)
ss2 = SubsplitSupport.from_tree(proj2)
sshat = ss1.mutualize(ss2)
sshat_trees = set(sshat.all_trees())
ps1 = PCSSSupport_old.from_tree(proj1)
ps2 = PCSSSupport_old.from_tree(proj2)
pshat = ps1.mutualize(ps2)
pshat_trees = set(pshat.all_trees())

tree in sshat_trees
tree in pshat_trees
len(sshat_trees)
len(pshat_trees)

sshat_trees == pshat_trees
for tree in sshat_trees: print(tree)

# Newick test 3

tree1 = MyTree("((((((A,E),D),C),F),B),G);")
print(tree1)
tree2 = MyTree("((((((A,D),E),B),F),C),G);")
print(tree2)
ss = SubsplitSupport.from_trees([tree1, tree2])
print(ss)
len(ss.all_trees())  # 4
ps = PCSSSupport_old.from_trees([tree1, tree2])
print(ps)
len(ps.all_trees())  # 2

rest1 = "ABCDEF"
rest2 = "ABCDEG"

trees = [tree1, tree2]
proj1s = [tree.restrict(rest1) for tree in trees]
proj2s = [tree.restrict(rest2) for tree in trees]
ss1 = SubsplitSupport.from_trees(proj1s)
ss2 = SubsplitSupport.from_trees(proj2s)
sshat = ss1.mutualize(ss2)
sshat_trees = set(sshat.all_trees())
ps1 = PCSSSupport_old.from_trees(proj1s)
ps2 = PCSSSupport_old.from_trees(proj2s)
pshat = ps1.mutualize(ps2)
pshat_trees = set(pshat.all_trees())

set(trees).issubset(sshat_trees)
set(trees).issubset(pshat_trees)
len(sshat_trees)  # 4
len(pshat_trees)  # 2

# DONE?: Find an example where PCSSSupport_old results in fewer virtual trees during mutualization

# Automation


def trees_restrict_reconstruct(taxon_set, k=10, ambig=1, n=2):
    restrictions = full_cover_restrictions_n(taxon_set, ambig, n)
    restriction = next(restrictions)
    rest1, rest2 = restriction

    trees = [MyTree.random(taxon_set) for _ in range(k)]
    proj1s = [tree.restrict(rest1) for tree in trees]
    proj2s = [tree.restrict(rest2) for tree in trees]
    ss1 = SubsplitSupport.from_trees(proj1s)
    ss2 = SubsplitSupport.from_trees(proj2s)
    sshat = ss1.mutualize(ss2)
    sshat_trees = set(sshat.all_trees())
    ps1 = PCSSSupport_old.from_trees(proj1s)
    ps2 = PCSSSupport_old.from_trees(proj2s)
    pshat = ps1.mutualize(ps2)
    pshat_trees = set(pshat.all_trees())

    return (SubsplitSupport.from_trees(trees).to_set().issubset(sshat.to_set()),
            PCSSSupport_old.from_trees(trees).to_set().issubset(pshat.to_set()),
            set(trees).issubset(sshat_trees), set(trees).issubset(pshat_trees),
            len(sshat_trees), len(pshat_trees), (trees, proj1s, proj2s))


X = "ABCDEFGHI"
subset_ss, subset_ps, subset_trees_ss, subset_trees_ps, ntrees_ss, ntrees_ps, details = trees_restrict_reconstruct(X, k=10, ambig=3)
print(f"{subset_ss}, {subset_ps}, {subset_trees_ss}, {subset_trees_ps}, {ntrees_ss}, {ntrees_ps}")

# Check a MyTree for a clade

tree = MyTree.random("ABCDE")
print(tree)

tree.check_clade("ABCDE")
tree.check_clade("AB")
tree.check_clade("ABC")

foo = tree.get_clade("ABE")
print(foo)

print(tree.get_clade("ABC"))

tree.check_subsplit(Subsplit("AB", "C"))
tree.check_subsplit(Subsplit("AB", "DE"))
tree.check_subsplit(Subsplit("AB", "C"))

# Check clade and subsplit probabilities

ccd = CCDSet.random_with_sparsity("ABCDEF", sparsity=0.5)
tree_dist = ccd.tree_distribution()

clade_probs = ccd.clade_probabilities()
clade_probs

clade = Clade("BCDF")
clade_probs[clade]
tree_dist.feature_prob(clade)

subsplit_probs = ccd.unconditional_probabilities()
subsplit_probs

subsplit = Subsplit("AD", "B")
subsplit_probs[subsplit]
tree_dist.feature_prob(subsplit)

# CCD stuff

reload(classes)
from classes import *

root_clade = Clade("ABCD")
ccd = CCDSet.random(root_clade)
tree_dist = ccd.tree_distribution()

c2c_probs = ccd.clade_to_clade_probabilities()
c_probs = ccd.clade_probabilities()

clade = Clade("ABD")
c_probs[clade]
c2c_probs[clade][root_clade]
c2c_probs[clade][clade]

parent_clade = Clade("ACD")
child_clade = Clade("CD")
c2c_probs[child_clade][parent_clade]
den = tree_dist.feature_prob(parent_clade)
num = tree_dist.prob_all([parent_clade, child_clade])
num / den

# Max probability tree and max clade credibility tree

reload(classes)
from classes import *

root_clade = Clade("ABCD")
ccd = CCDSet.random(root_clade)
tree_dist = ccd.tree_distribution()

print(tree_dist)
best_tree = tree_dist.max_item()
best_lik = tree_dist.max_likelihood()
print(best_tree)
print(best_lik)
ccd_best_tree, ccd_best_lik = ccd.max_prob_tree()
print(ccd_best_tree)
print(ccd_best_lik)

# print(ccd)

ccd = CCDSet({Subsplit("ABC", "D"): 0.40, Subsplit("A", "BCD"): 0.60, Subsplit("AB", "C"): 1.0, Subsplit("A", "B"): 1.0,
              Subsplit("BC", "D"): 0.33, Subsplit("BD", "C"): 0.33, Subsplit("B", "CD"): 0.34,
              Subsplit("B", "C"): 1.0, Subsplit("B", "D"): 1.0, Subsplit("C", "D"): 1.0})
ccd.normalize()
tree_dist = ccd.tree_distribution()
print(tree_dist)
best_tree = tree_dist.max_item()
best_lik = tree_dist.max_likelihood()
print(best_tree)
print(best_lik)
ccd_best_tree, ccd_best_lik = ccd.max_prob_tree()
print(ccd_best_tree)
print(ccd_best_lik)

clade_probs = ccd.clade_probabilities()
clade_probs

max_clade_cred_tree = max(tree_dist, key=lambda tree: tree.clade_score(clade_probs))
print(max_clade_cred_tree)
print(max_clade_cred_tree.clade_score(clade_probs))
{tree: tree.clade_score(clade_probs) for tree in tree_dist}

max_clade_tree, max_clade_score = ccd.max_clade_tree()
print(max_clade_tree)
print(max_clade_score)

# Clade-to-clade stress test

reload(classes)
from classes import *

root_clade = Clade("ABCDEFGHI")
ccd = CCDSet.random(root_clade, concentration=2, cutoff=0.0)
print(ccd)
len(ccd)

c_probs = ccd.clade_probabilities()
# c_probs2 = ccd.clade_probabilities_old()
c2c_probs = ccd.clade_to_clade_probabilities()

# all(abs(c_probs[key] - c_probs2[key]) < 1e-14 for key in c_probs)

# %timeit ccd.clade_probabilities()
# %timeit ccd.clade_probabilities_old()
# %timeit ccd.clade_to_clade_probabilities()

# Max probability tree stress test

reload(classes)
from classes import *

root_clade = Clade("ABCDEFGHIJK")
ccd = CCDSet.random(root_clade, concentration=2, cutoff=0.001)
print(ccd)
len(ccd)

tree_dist = ccd.tree_distribution()

# %timeit tree_dist = ccd.tree_distribution()
# %timeit tree_dist.max_item()
# %timeit ccd.tree_distribution().max_item()

# %timeit ccd.highest_prob_tree()
# %timeit ccd.max_clade_tree()

# Restriction summarization experiment

reload(classes)
from classes import *

root_clade = Clade("ABCDEFG")
ccd = CCDSet.random(root_clade)
tree_dist = ccd.tree_distribution()

restriction = Clade("ABCDE")
tree_dist_restricted = tree_dist.restrict(restriction)

clade = Clade("BCDE")
all_reductions = [clade | set(ps) for ps in powerset(root_clade - restriction)]

tree_dist_restricted.feature_prob(clade)  # Setting the bar

sum(tree_dist.feature_prob(reduction) for reduction in all_reductions)  # Too high!
tree_dist.prob_any(all_reductions)  # Just right!

ccd_restricted = ccd.restrict(restriction)
ccd_restricted.clade_probabilities()[clade]  # Just right!

subsplit = Subsplit("BC", "DE")
tree_dist_restricted.feature_prob(subsplit)  # Setting the bar A

tree_dist_restricted.feature_prob(subsplit.clade())  # Setting the bar 3
tree_dist_restricted.feature_prob(subsplit) / tree_dist_restricted.feature_prob(subsplit.clade())  # Setting the bar 2
ccd_restricted[subsplit]  # Just right 2

uncond_probs = ccd.unconditional_probabilities()
uncond_probs_restricted = ccd_restricted.unconditional_probabilities()

uncond_probs_restricted[subsplit]  # Just right! A
sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction) == subsplit)  # Just right! A

sum(sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction) == sss) for sss in uncond_probs_restricted if sss.clade() == subsplit.clade())  # Just right 3
sum(uncond_probs_restricted[ss] for ss in uncond_probs_restricted if ss.clade() == subsplit.clade())  # Just right 3
uncond_probs_restricted[subsplit] / sum(uncond_probs_restricted[ss] for ss in uncond_probs_restricted if ss.clade() == subsplit.clade())  # Just right 2
sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction) == subsplit) / sum(sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction) == sss) for sss in uncond_probs if sss.restrict(restriction).clade() == subsplit.clade())  # Just right 2

tree_dist_restricted.feature_prob(clade)  # Setting the bar
sum(uncond_probs[ss] for ss in uncond_probs if clade in ss.restrict(restriction))  # Too high!
sum(uncond_probs_restricted[ss] for ss in uncond_probs_restricted if clade in ss)  # Just right!

sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction).clade() == clade)

a = sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction) == subsplit)
b = sum(uncond_probs[ss] for ss in uncond_probs if ss.restrict(restriction).clade() == subsplit.clade())
a/b
ccd_restricted[subsplit]

# New ProbabilityDistribution tests (formerly called ProbabilityDistribution2 here)
from importlib import reload
import classes
reload(classes)
from classes import *

foo = ProbabilityDistribution()
foo.set_lin('bar', 0.2)
foo.params

foo.set_lin('baz', 0.3)
foo.params

bar = foo.copy()
foo.normalize()
bar.params
foo.params
foo.probs()
str(bar)

foo.add_log('bar', 1)
foo.probs()
foo.normalize()
foo.probs()

# Gradient experiments

reload(classes)
from classes import *

ccd = CCDSet({Subsplit("ABC", "D"): 0.40, Subsplit("A", "BCD"): 0.60, Subsplit("AB", "C"): 1.0, Subsplit("A", "B"): 1.0,
              Subsplit("BC", "D"): 0.33, Subsplit("BD", "C"): 0.33, Subsplit("B", "CD"): 0.34,
              Subsplit("B", "C"): 1.0, Subsplit("B", "D"): 1.0, Subsplit("C", "D"): 1.0})
ccd.normalize()

ccd.unconditional_probabilities()

ccd2 = ccd.copy()
# ccd2.unconditional_probabilities()

clade = Clade("BCD")
subsplit = Subsplit("BC","D")
delta = 0.0001
ccd2[clade].add_log(subsplit, delta)
ccd2.normalize(clade)

uncond = ccd.unconditional_probabilities()
uncond2 = ccd2.unconditional_probabilities()

theo_deriv = uncond[subsplit]*(1.0 - ccd[subsplit])
est_deriv = (uncond2[subsplit] - uncond[subsplit])/delta
print(theo_deriv)
print(est_deriv)

## Experiment 2

ccd = CCDSet({Subsplit("ABC", "D"): 0.40, Subsplit("A", "BCD"): 0.60, Subsplit("AB", "C"): 1.0, Subsplit("A", "B"): 1.0,
              Subsplit("BC", "D"): 0.33, Subsplit("BD", "C"): 0.33, Subsplit("B", "CD"): 0.34,
              Subsplit("B", "C"): 1.0, Subsplit("B", "D"): 1.0, Subsplit("C", "D"): 1.0})
ccd.normalize()

ccd2 = ccd.copy()

subsplit1 = Subsplit("A","BCD")
clade1 = subsplit1.clade()
subsplit2 = Subsplit("B","CD")
clade2 = subsplit2.clade()
delta = 0.0001
ccd2[clade1].add_log(subsplit1, delta)
ccd2.normalize(clade1)

uncond = ccd.unconditional_probabilities()
c2c = ccd.clade_to_clade_probabilities()
uncond2 = ccd2.unconditional_probabilities()

theo_deriv = uncond[subsplit1]*(c2c[clade2][clade2] - c2c[clade2][clade1])*ccd[subsplit2]
est_deriv = (uncond2[subsplit2] - uncond[subsplit2])/delta
print(theo_deriv)
print(est_deriv)

## Experiment 3

ccd = CCDSet.random("ABCDEF")
delta = 0.0001
subsplit1 = Subsplit("A","BCDE")
subsplit2 = Subsplit("B","CD")

clade1 = subsplit1.clade()
clade2 = subsplit2.clade()
subsplit1_ch = subsplit1.compatible_child(subsplit2)

ccd2 = ccd.copy()
ccd2[clade1].add_log(subsplit1, delta)
ccd2.normalize(clade1)

uncond = ccd.unconditional_probabilities()
c2c = ccd.clade_to_clade_probabilities()
uncond2 = ccd2.unconditional_probabilities()

theo_deriv = uncond[subsplit1]*(c2c[clade2][subsplit1_ch] - c2c[clade2][clade1])*ccd[subsplit2]
est_deriv = (uncond2[subsplit2] - uncond[subsplit2])/delta
print(theo_deriv)
print(est_deriv)

# General unconditional probability gradient experiments

reload(classes)
from classes import *



# def uncond_prob_derivative(prob_of: Subsplit, wrt: Subsplit, ccd: CCDSet, c2c: dict=None):
#     root_clade = ccd.root_clade()
#     clade1 = wrt.clade()
#     clade2 = prob_of.clade()
#     if not clade2.issubset(clade1):
#         return 0.0
#     if c2c is None:
#         c2c = ccd.clade_to_clade_probabilities()
#     uncond_wrt = c2c.get(clade1, dict()).get(root_clade, 0.0) * ccd[wrt]
#     if clade1 == clade2:
#         indic = 1.0 if (prob_of == wrt) else 0.0
#         return uncond_wrt*(indic - ccd[wrt])
#     comp_child = wrt.compatible_child(prob_of)
#     comp_child_to_clade2 = 0.0
#     if comp_child is not None:
#         comp_child_to_clade2 = c2c.get(clade2, dict()).get(comp_child, 0.0)
#     clade1_to_clade2 = c2c.get(clade2, dict()).get(clade1, 0.0)
#     return uncond_wrt*(comp_child_to_clade2 - clade1_to_clade2)*ccd[prob_of]


ccd = CCDSet.random("ABCDEF")
delta = 0.0001

c2c = ccd.clade_to_clade_probabilities()

clade1 = random.choice(list(ccd.clades()))
wrt = random.choice(list(ccd[clade1].keys()))
clade2 = random.choice(list(ccd.clades()))
prob_of = random.choice(list(ccd[clade2].keys()))

est_deriv = estimate_derivative(prob_of, wrt, ccd, delta=delta)
theo_deriv = ccd.uncond_prob_derivative(prob_of, wrt, c2c)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

## Drawn from c2c probs

ccd = CCDSet.random("ABCDEF")
delta = 0.00001

c2c = ccd.clade_to_clade_probabilities()

clade2 = random.choice(list(c2c.keys()))
prob_of = random.choice(list(ccd[clade2].keys()))
clade1 = random.choice(list(c2c[clade2].keys()))
wrt = random.choice(list(ccd[clade1].keys()))

est_deriv = estimate_derivative(prob_of, wrt, ccd, delta=delta)
theo_deriv = ccd.uncond_prob_derivative(prob_of, wrt, c2c)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

# Gradient check under restriction



# def restricted_uncond_prob_derivative(restriction, prob_of: Subsplit, wrt: Subsplit, ccd: CCDSet, c2c: dict=None):
#     restriction = Clade(restriction)
#     if c2c is None:
#         c2c = ccd.clade_to_clade_probabilities()
#     result = 0.0
#     for subsplit in ccd.iter_subsplits():
#         # print(f"Examining subsplit {subsplit}:")
#         if subsplit.restrict(restriction) == prob_of:
#             # print("Found a match")
#             result += ccd.uncond_prob_derivative(prob_of=subsplit, wrt=wrt, c2c=c2c)
#     return result


X = "ABCDEFG"
Xbar = "ABCDE"
prob_of = Subsplit("BC", "D")
wrt = Subsplit("AE", "BCDF")

ccd = CCDSet.random(X)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()

ccd_r = ccd.restrict(Xbar)
uncond_r = ccd_r.unconditional_probabilities()
uncond_r[prob_of]

ccd2 = ccd.copy()
ccd2[wrt.clade()].add_log(wrt, delta)
ccd2.normalize()
ccd2_r = ccd2.restrict(Xbar)
uncond2_r = ccd2_r.unconditional_probabilities()
(uncond2_r[prob_of] - uncond_r[prob_of]) / delta

est_deriv = estimate_restricted_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd, delta=delta)
theo_deriv = ccd.restricted_uncond_prob_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, c2c=c2c)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist_r = ccd_r.tree_distribution()
tree_dist2_r = ccd2_r.tree_distribution()
tree_dist_r.feature_prob(prob_of)
uncond_r[prob_of]
tree_dist2_r.feature_prob(prob_of)
uncond2_r[prob_of]



# def restricted_clade_prob_derivative(restriction, prob_of: Clade, wrt: Subsplit, ccd: CCDSet, c2c: dict=None):
#     restriction = Clade(restriction)
#     if c2c is None:
#         c2c = ccd.clade_to_clade_probabilities()
#     result = 0.0
#     for subsplit in ccd.iter_subsplits():
#         # print(f"Examining subsplit {subsplit}:")
#         if subsplit.restrict(restriction).clade() == prob_of and not subsplit.restrict(restriction).is_trivial():
#             # print("Found a match")
#             result += ccd.uncond_prob_derivative(prob_of=subsplit, wrt=wrt, c2c=c2c)
#     return result


X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")
prob_of = Clade("BCD")
wrt = Subsplit("AE", "BCDF")

ccd = CCDSet.random(X)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()

ccd_r = ccd.restrict(Xbar)
uncond_r = ccd_r.clade_probabilities()
uncond_r[prob_of]

ccd2 = ccd.copy()
ccd2[wrt.clade()].add_log(wrt, delta)
ccd2.normalize()
ccd2_r = ccd2.restrict(Xbar)
uncond2_r = ccd2_r.clade_probabilities()
(uncond2_r[prob_of] - uncond_r[prob_of]) / delta

est_deriv = estimate_restricted_clade_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd, delta=delta)
theo_deriv = ccd.restricted_clade_prob_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd, c2c=c2c)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist_r = ccd_r.tree_distribution()
tree_dist2_r = ccd2_r.tree_distribution()
tree_dist_r.feature_prob(prob_of)
uncond_r[prob_of]
tree_dist2_r.feature_prob(prob_of)
uncond2_r[prob_of]

subsplit_uncond = ccd.unconditional_probabilities()
subsplit_uncond_r = ccd_r.unconditional_probabilities()
sum(sum(subsplit_uncond[ss] for ss in subsplit_uncond if ss.restrict(Xbar) == sss) for sss in subsplit_uncond_r if sss.clade() == prob_of)
sum(sum(subsplit_uncond[ss] for ss in subsplit_uncond if ss.restrict(Xbar) == sss) for sss in subsplit_uncond if sss.clade() & Xbar == prob_of)
sum(sum(subsplit_uncond[ss] for ss in subsplit_uncond if ss.restrict(Xbar) == sss) for sss in subsplit_uncond if sss.restrict(Xbar).clade() == prob_of)
sum(subsplit_uncond_r[ss] for ss in subsplit_uncond_r if ss.clade() == prob_of)
uncond_r[prob_of]
restricted_subsplits = {ss.restrict(Xbar) for ss in subsplit_uncond if ss.restrict(Xbar).clade() == prob_of and not ss.restrict(Xbar).is_trivial()}
sum(sum(subsplit_uncond[ss] for ss in subsplit_uncond if ss.restrict(Xbar) == sss) for sss in restricted_subsplits if sss.clade() == prob_of)
sum(sum(subsplit_uncond[ss] for ss in subsplit_uncond if ss.restrict(Xbar) == sss) for sss in
    {ssss.restrict(Xbar) for ssss in subsplit_uncond if ssss.restrict(Xbar).clade() == prob_of and not ssss.restrict(Xbar).is_trivial()})
sum(sum(subsplit_uncond[ss] for ss in subsplit_uncond if ss.restrict(Xbar) == sss) for sss in
    {ssss.restrict(Xbar) for ssss in subsplit_uncond if ssss.restrict(Xbar).clade() == prob_of})  # Bad (on purpose)
tree_dist_r.feature_prob(prob_of)

# Restricted conditional probability derivative experiments



X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")
prob_of = Subsplit("A", "BCD")
wrt = Subsplit("ABCD", "F")

ccd = CCDSet.random(X)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()

ccd_r = ccd.restrict(Xbar)

ccd2 = ccd.copy()
ccd2[wrt.clade()].add_log(wrt, delta)
ccd2.normalize()
ccd2_r = ccd2.restrict(Xbar)
(ccd2_r[prob_of] - ccd_r[prob_of]) / delta

est_deriv = estimate_restricted_conditional_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd, delta=delta)
theo_deriv = ccd.restricted_cond_prob_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, c2c=c2c, restricted_self=ccd_r)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

# KL-divergence derivative experiments

X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")
# wrt = Subsplit("A", "BC")
ccd = CCDSet.random(X)
ccd_small = CCDSet.random(Xbar)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()

ccd_r = ccd.restrict(Xbar)
ccd_small.kl_divergence(ccd_r)

# clade1 = random.choice(list(ccd.clades()))
wrt = random.choice(list(ccd.support()))

ccd2 = ccd.copy()
ccd2[wrt.clade()].add_log(wrt, delta)
ccd2.normalize()
ccd2_r = ccd2.restrict(Xbar)
(ccd_small.kl_divergence(ccd2_r) - ccd_small.kl_divergence(ccd_r)) / delta

est_deriv = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
theo_deriv = ccd.restricted_kl_divergence_derivative_old(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r)
# theo_deriv = ccd.restricted_kl_divergence_derivative(other=ccd_small, wrt=wrt)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tdist_small = ccd_small.tree_distribution()
tdist_r = ccd_r.tree_distribution()
tdist_small.kl_divergence(tdist_r)
ccd_small.kl_divergence(ccd_r)

tdist2_r = ccd2_r.tree_distribution()
tdist_small.kl_divergence(tdist2_r)
ccd_small.kl_divergence(ccd2_r)
(tdist_small.kl_divergence(tdist2_r) - tdist_small.kl_divergence(tdist_r)) / delta

# Looking for bad derivatives experiments

X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")

ccd = CCDSet.random(X)
ccd_small = CCDSet.random(Xbar)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()
ccd_r = ccd.restrict(Xbar)
ccd_small.kl_divergence(ccd_r)

# clade1 = random.choice(list(ccd.clades()))
# wrt = random.choice(list(ccd[clade1].keys()))
wrt = Subsplit("ABC", "EF")

verbose_dict = dict()
est_deriv = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
theo_deriv = ccd.restricted_kl_divergence_derivative_old(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r, verbose_dict=verbose_dict)
for sbar in verbose_dict:
    est_cond = estimate_restricted_conditional_derivative(Xbar, sbar, wrt, ccd, delta)
    verbose_dict[sbar]['est'] = est_cond

for sbar in verbose_dict:
    if len(sbar.clade()) >= 3:
        # print(f"{sbar}:\tP={verbose_dict[sbar]['P']:10.4g}, q={verbose_dict[sbar]['q']:10.4g}, dq={verbose_dict[sbar]['dq']:10.4g}, est={verbose_dict[sbar]['est']:10.4g}, diff={abs(verbose_dict[sbar]['dq']-verbose_dict[sbar]['est']):10.4g}, rel={abs(verbose_dict[sbar]['dq']-verbose_dict[sbar]['est'])/max(abs(verbose_dict[sbar]['dq']), abs(verbose_dict[sbar]['est'])):10.4g}")
        print(f"{sbar}:\tP={verbose_dict[sbar]['P']:10.4g}, q={verbose_dict[sbar]['q']:10.4g}, dq={verbose_dict[sbar]['dq']:10.4g}, est={verbose_dict[sbar]['est']:10.4g}, diff={abs(verbose_dict[sbar]['dq'] - verbose_dict[sbar]['est']):10.4g}")

chosen_sss = {ss for ss in ccd_r.iter_subsplits() if len(ss.clade()) == 5}
for ss in chosen_sss:
    est_uncond_deriv = estimate_restricted_derivative(Xbar, ss, wrt, ccd, delta)
    theo_uncond_deriv = ccd.restricted_uncond_prob_derivative(Xbar, ss, wrt, c2c)
    print(f"dQ({ss}): est={est_uncond_deriv:10.4g}, theo={theo_uncond_deriv:10.4g}, rel={abs(est_uncond_deriv-theo_uncond_deriv)/abs(theo_uncond_deriv):10.4g}")

# Update manually
manual_list = [Subsplit("B", "CE"), Subsplit("BE", "C"), Subsplit("BC", "E")]
manual_result = 0.0
for ss in manual_list:
    manual_result += -verbose_dict[ss]['P']*verbose_dict[ss]['dq']/verbose_dict[ss]['q']

# Update manually
manual_list = [Subsplit("B", "CE"), Subsplit("BE", "C"), Subsplit("BC", "E")]
for sbar in manual_list:
    est_deriv1 = estimate_restricted_conditional_derivative(Xbar, sbar, wrt, ccd, delta)
    theo_deriv1 = ccd.restricted_cond_prob_derivative(Xbar, sbar, wrt, c2c, ccd_r)
    print(f"est={est_deriv1:10.4g}, theo={theo_deriv1:10.4g}, diff={abs(est_deriv1-theo_deriv1):10.4g}")


ccd2 = ccd.copy()
ccd2[wrt.clade()].add_log(wrt, delta)
ccd2.normalize()
ccd2_r = ccd2.restrict(Xbar)
(ccd_small.kl_divergence(ccd2_r) - ccd_small.kl_divergence(ccd_r)) / delta

## Check for bad derivatives

X = Clade("ABCDEFG")
ccd = CCDSet.random(X)
c2c = ccd.clade_to_clade_probabilities()
delta = 0.000001
wrt = Subsplit("ABC", "EF")

verbose_dict = dict()
for ss in ccd.iter_subsplits():
    est_uncond_deriv = estimate_derivative(ss, wrt, ccd, delta)
    theo_uncond_deriv = ccd.uncond_prob_derivative(ss, wrt, c2c)
    abs_diff = abs(est_uncond_deriv - theo_uncond_deriv)
    print(f"{ss}: \t est={est_uncond_deriv:10.4g}, theo={theo_uncond_deriv:10.4g}, diff={abs_diff:10.4g}")

# Test new KL derivative function

reload(classes)
from classes import *

X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")
# wrt = Subsplit("A", "BC")
ccd = CCDSet.random(X)
ccd_small = CCDSet.random(Xbar)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()

ccd_r = ccd.restrict(Xbar)
ccd_small.kl_divergence(ccd_r)

# clade1 = random.choice(list(ccd.clades()))
wrt = random.choice(list(ccd.support()))

ccd2 = ccd.copy()
ccd2[wrt.clade()].add_log(wrt, delta)
ccd2.normalize()
ccd2_r = ccd2.restrict(Xbar)
(ccd_small.kl_divergence(ccd2_r) - ccd_small.kl_divergence(ccd_r)) / delta

est_deriv = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
theo_deriv = ccd.restricted_kl_divergence_derivative_old(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r)
theo_deriv2 = ccd.restricted_kl_divergence_derivative(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv2 - est_deriv))
print(abs(theo_deriv - theo_deriv2))
print(abs(theo_deriv2 - est_deriv)/abs(est_deriv))

# %timeit estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
# %timeit ccd.restricted_kl_divergence_derivative_old(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r)
# %timeit ccd.restricted_kl_divergence_derivative(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r)

# Test new KL gradient function

reload(classes)
from classes import *

X = Clade("ABCDEFG")
Xbar = Clade("ABCDE")

ccd = CCDSet.random(X)
ccd_small = CCDSet.random(Xbar)
delta = 0.000001

c2c = ccd.clade_to_clade_probabilities()

ccd_r = ccd.restrict(Xbar)
ccd_small.kl_divergence(ccd_r)

theo_grad = ccd.restricted_kl_divergence_gradient(other=ccd_small, c2c=c2c, restricted_self=ccd_r)

wrt = random.choice(list(theo_grad.keys()))
est_deriv = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
print(abs(theo_grad[wrt] - est_deriv))
print(abs(theo_grad[wrt] - est_deriv)/abs(est_deriv) if est_deriv != 0.0 else -abs(theo_grad[wrt] - est_deriv))

# %timeit ccd.restricted_kl_divergence_gradient(other=ccd_small, c2c=c2c, restricted_self=ccd_r)

for wrt_temp in theo_grad:
    est_deriv_temp = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt_temp, delta=delta)
    print(f"{wrt_temp}: \t grad={theo_grad[wrt_temp]:10.4g} \t est={est_deriv_temp:10.4g} \t diff={abs(theo_grad[wrt_temp]-est_deriv_temp):10.4g}")
