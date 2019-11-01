from small_trees import *

from collections import defaultdict
import itertools

# Experiment 0

single_tree_manifests = generate_unrooted("ABCD")
for tree in single_tree_manifests: print(tree)

single_tree_manifests = generate_rooted("ABCD")
for tree in single_tree_manifests: print(tree)

# Experiment 1

X = "ABCD"
all_trees_iter = generate_rooted(X)
X1 = "ABC"
X2 = "ABD"

tree_count = Counter()
for tree in all_trees_iter:
    projection1 = restrict_tree(tree, X1)
    projection2 = restrict_tree(tree, X2)
    tree_count[projection1.get_topology_id(), projection2.get_topology_id()] += 1
tree_count
len(tree_count)

# Experiment 2

X = "ABCDEFG"
restrictions = combinations(X, 4)

counters = explore_rooted_trees(X, restrictions)

for key in counters: print(counters[key].values())

for key in counters: print(Counter(counters[key].values()))

Counter(frozenset(Counter(counters[key].values()).items()) for key in counters)

# Experiment 3

X = "ABCD"
ambig = 1
# restrictions = combinations(combinations(X, len(X)-ambig), 2)
restrictions = full_cover_restrictions(X, ambig)
restriction = next(restrictions)
# restriction = ("ABC", "ABD")
print(restriction)

result = visualize_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i = 4
print(result[result_keys[i]].proj1)
print(result[result_keys[i]].proj2)
for tree in result[result_keys[i]].treelist: print(tree)

for tree in result[result_keys[i]].treelist: print(get_subsplits(tree))

for tree in result[result_keys[i]].treelist: print(restrict_tree(tree, "ABD"))

full_subsplit_set = set()
for tree in result[result_keys[i]].treelist:
    full_subsplit_set |= set(get_subsplits(tree))
print(full_subsplit_set)

# Experiment 4

X = "ABCDEF"
ambig = 2
restrictions = full_cover_restrictions(X, ambig)
leaf_set1, leaf_set2 = restriction = next(restrictions)
# restriction = ("ABC", "ABD")

result = visualize_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

# for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

# Find all valid n_trees values
unique_trees = Counter()
for i, key in enumerate(result_keys):
    unique_trees[len(result[key].treelist)] += 1
print(unique_trees)

n_trees = 1
signatures = Counter()
print("Index\tTrees\tUnion\tAgree\t 1-2 \t 2-1 \tSubsplits")
for i, key in enumerate(result_keys):
    if len(result[key].treelist) == n_trees:
        subsplit1 = get_subsplits(result[result_keys[i]].proj1, leaf_set2)
        subsplit2 = get_subsplits(result[result_keys[i]].proj2, leaf_set1)
        subsplit_counter1 = Counter(subsplit1)
        subsplit_counter2 = Counter(subsplit2)
        sig = (sum((subsplit_counter1 | subsplit_counter2).values()),
               sum((subsplit_counter1 & subsplit_counter2).values()),
               sum((subsplit_counter1 - subsplit_counter2).values()),
               sum((subsplit_counter2 - subsplit_counter1).values()))
        signatures[sig] += 1
        print(f"{i:^5}\t{len(result[key].treelist):^5}\t{sum((subsplit_counter1 | subsplit_counter2).values()):^5}\t{sum((subsplit_counter1 & subsplit_counter2).values()):^5}\t{sum((subsplit_counter1 - subsplit_counter2).values()):^5}\t{sum((subsplit_counter2 - subsplit_counter1).values()):^5}\t{subsplit1}\t{subsplit2}")
print(signatures)

# TODO: Brainstorm ways to summarize the similarity and differences between two lists/multisets
single_tree_manifests = Counter([1, 1, 2])
bar = Counter([1,2,3])
single_tree_manifests | bar
single_tree_manifests & bar
sum((single_tree_manifests | bar).values())
sum((single_tree_manifests & bar).values())

i=3
print(result[result_keys[i]].proj1)
print(result[result_keys[i]].proj2)
# for tree in result[result_keys[i]].treelist: print(tree)

get_subsplits(result[result_keys[i]].proj1)
print(get_subsplits(result[result_keys[i]].proj1, leaf_set2))

get_subsplits(result[result_keys[i]].proj2)
print(get_subsplits(result[result_keys[i]].proj2, leaf_set1))

subsplit_set1 = set(get_subsplits(result[result_keys[i]].proj1, leaf_set2))
subsplit_set2 = set(get_subsplits(result[result_keys[i]].proj2, leaf_set1))
print(subsplit_set1)
print(subsplit_set2)
len(subsplit_set1 | subsplit_set2)
print(subsplit_set1 | subsplit_set2)

# Experiment 5

X = "ABCDEF"
ambig = 2
restrictions = full_cover_restrictions(X, ambig)
leaf_set1, leaf_set2 = restriction = next(restrictions)
# restriction = ("ABC", "ABD")

result = visualize_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

sig_to_n_trees = defaultdict(Counter)
n_trees_to_sig = defaultdict(Counter)
n_trees_sig_subsplits = defaultdict(lambda: defaultdict(Counter))
print("Index\tTrees\tUnion\tAgree\t 1-2 \t 2-1 \tSubsplits")
for i, key in enumerate(result_keys):
    n_trees = len(result[key].treelist)
    subsplit1 = get_subsplits(result[result_keys[i]].proj1, leaf_set2)
    subsplit2 = get_subsplits(result[result_keys[i]].proj2, leaf_set1)
    subsplit_counter1 = Counter(subsplit1)
    subsplit_counter2 = Counter(subsplit2)
    sig = (sum((subsplit_counter1 | subsplit_counter2).values()),
           sum((subsplit_counter1 & subsplit_counter2).values()),
           sum((subsplit_counter1 - subsplit_counter2).values()),
           sum((subsplit_counter2 - subsplit_counter1).values()))
    sig_to_n_trees[sig][n_trees] += 1
    n_trees_to_sig[n_trees][sig] += 1
    n_trees_sig_subsplits[n_trees][sig][(subsplit1, subsplit2)] += 1
    print(f"{i:^5}\t{n_trees:^5}\t{sum((subsplit_counter1 | subsplit_counter2).values()):^5}\t{sum((subsplit_counter1 & subsplit_counter2).values()):^5}\t{sum((subsplit_counter1 - subsplit_counter2).values()):^5}\t{sum((subsplit_counter2 - subsplit_counter1).values()):^5}\t{subsplit1}\t{subsplit2}")
sig_to_n_trees
n_trees_to_sig
n_trees_sig_subsplits[3][(4,4,0,0)]

# Experiment 6

# get_compatible_subsplits({{},{}}, {{},{}})

get_compatible_subsplits({{'A'},{'B','C'}}, {{'A', 'D'},{'E'}})

# Experiment 7

X = "ABCDEF"
ambig = 2
restrictions = full_cover_restrictions(X, ambig)
leaf_set1, leaf_set2 = restriction = next(restrictions)

result = visualize_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i = 222
print(result[result_keys[i]].proj1)
print(result[result_keys[i]].proj2)
for tree in result[result_keys[i]].treelist: print(tree)

for tree in result[result_keys[i]].treelist: print(get_subsplits(tree))

empirical_support_string_set = set()
for tree in result[result_keys[i]].treelist:
    subsplits = get_subsplits(tree)
    for subsplit in subsplits:
        clade_string = subsplit.replace(':', '')
        if len(clade_string) > 2:
            empirical_support_string_set.add(subsplit)

tree1 = result[result_keys[i]].proj1
tree2 = result[result_keys[i]].proj2
support = generate_support(tree1, tree2, verbose=True)

print(sorted(empirical_support_string_set))
print(sorted(support_to_string_set(support)))

# Experiment 8

X = "ABCDEF"
all_trees = generate_rooted(X)
big_tree = next(all_trees)
leaf_set1, leaf_set2, leaf_set3 = "ABC", "CDE", "AEF"

tree1 = restrict_tree(big_tree, leaf_set1)
tree2 = restrict_tree(big_tree, leaf_set2)
tree3 = restrict_tree(big_tree, leaf_set3)
print(tree1)
print(tree2)
print(tree3)

manifest1 = get_subsplit_manifest(tree1)
manifest2 = get_subsplit_manifest(tree2)
manifest3 = get_subsplit_manifest(tree3)

manifest12 = generate_mutual_manifest(manifest1, manifest2, verbose=True)
support12 = generate_support(tree1, tree2, verbose=True)

support_to_string_set(support12)
# TODO: This will come in handy for writing a manifest-prettifier
for subsplit_list in manifest12.values(): print(support_to_string_set(subsplit_list))

manifest123 = generate_mutual_manifest(manifest12, manifest3, verbose=True)
# TODO: This errors currently
# candidate_subsplits1 = manifest1[restricted_clade1]
# KeyError: frozenset({'E', 'A'})
for subsplit_list in manifest123.values(): print(support_to_string_set(subsplit_list))
print(big_tree)

# Experiment 9

X = "ABCDEF"
ambig = 2
n = 3
restrictions = full_cover_restrictions_n(X, ambig, n)
restriction = next(restrictions)

result = visualize_restricted_rooted_trees_n(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i = 359
for tree in result[result_keys[i]].projections: print(tree)
# for tree in result[result_keys[i]].treelist: print(tree)
for tree in result[result_keys[i]].treelist:
    big_manifest = get_subsplit_manifest(tree)
    print(tree)
    print_manifest(big_manifest)

big_manifests = [get_subsplit_manifest(tree) for tree in result[result_keys[i]].treelist]

projection_manifests = [get_subsplit_manifest(tree) for tree in result[result_keys[i]].projections]

mutual_manifest = generate_mutual_manifest_n(projection_manifests)
print_manifest(mutual_manifest)

merged_manifest = merge_manifests(big_manifests)
print_manifest(merged_manifest)

print(merged_manifest == mutual_manifest)

for i, key in enumerate(result_keys):
    big_trees = result[result_keys[i]].treelist
    projections = result[result_keys[i]].projections
    big_manifests = [get_subsplit_manifest(tree) for tree in big_trees]
    projection_manifests = [get_subsplit_manifest(tree) for tree in projections]
    merged_manifest = merge_manifests(big_manifests)
    mutual_manifest = generate_mutual_manifest_n(projection_manifests)
    print(f"{i}: {merged_manifest == mutual_manifest}")

# Experiment 10

X = "ABCDE"
ambig = 1
n = 2
restrictions = full_cover_restrictions_n(X, ambig, n)
restriction = next(restrictions)

result = visualize_restricted_rooted_trees_n(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i1 = 14
pjts1 = result[result_keys[i1]]
big_tree1 = pjts1.treelist[0]
projection1 = pjts1.projections[0]
for tree in pjts1.treelist: print(tree)
for proj in pjts1.projections: print(proj)

i2 = 46
pjts2 = result[result_keys[i2]]
big_tree2 = pjts2.treelist[0]
projection2 = pjts2.projections[1]
for tree in pjts2.treelist: print(tree)
for proj in pjts2.projections: print(proj)

print(big_tree1)
print(big_tree2)

print(projection1)
print(projection2)

manifest1 = get_subsplit_manifest(projection1)
manifest2 = get_subsplit_manifest(projection2)

manifest12 = generate_mutual_manifest(manifest1, manifest2, verbose=True)

# Experiment 11

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

single_tree_manifests = possible_single_tree_manifests(manifest12, verbose=True)

j = 2
print_manifest(single_tree_manifests[j], show_degenerate=False)
print(manifest_to_tree(single_tree_manifests[j]))

for manifest in single_tree_manifests: print(manifest_to_tree(manifest))
for tree in big_tree1: print(tree)

# Experiment 12

X = "ABCDE"
ambig = 1
n = 2
restrictions = full_cover_restrictions_n(X, ambig, n)
restriction = next(restrictions)

result = visualize_restricted_rooted_trees_n(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

## Incompatible, singletons per each restriction signature
i1 = [5, 25, 67]
i2 = [19, 43, 73]

## Compatible, singletons
i1 = [10, 34, 59]
i2 = [10, 34, 59]

pjts1 = [result[result_keys[i]] for i in i1]
big_trees1 = [pjt.treelist for pjt in pjts1]
projections1 = [pjt.projections[0] for pjt in pjts1]
for i, big_tree1, proj1 in zip(i1, big_trees1, projections1):
    print(f"# Signature {i}")
    print("Joint trees:")
    for big_tree in big_tree1:
        print(big_tree)
    print(f"Projection on {''.join(restriction[0])}")
    print(proj1)

pjts2 = [result[result_keys[i]] for i in i2]
big_trees2 = [pjt.treelist for pjt in pjts2]
projections2 = [pjt.projections[1] for pjt in pjts2]
for i, big_tree2, proj2 in zip(i2, big_trees2, projections2):
    print(f"# Signature {i}")
    print("Joint trees:")
    for big_tree in big_tree2:
        print(big_tree)
    print(f"Projection on {''.join(restriction[1])}")
    print(proj2)


manifests1 = [get_subsplit_manifest(proj) for proj in projections1]
manifests2 = [get_subsplit_manifest(proj) for proj in projections2]

merged_manifest1 = merge_manifests(manifests1)
merged_manifest2 = merge_manifests(manifests2)

for manifest in manifests1: print(manifest_to_tree(manifest))
reconstructions1 = possible_single_tree_manifests(merged_manifest1)
for reconstruction in reconstructions1: print(manifest_to_tree(reconstruction))

manifest12 = generate_mutual_manifest(merged_manifest1, merged_manifest2, verbose=True)
print_manifest(manifest12, show_degenerate=False)

single_tree_manifests = possible_single_tree_manifests(manifest12, verbose=True)

for manifest in single_tree_manifests: print(manifest_to_tree(manifest))
for trees in itertools.chain(big_trees1, big_trees2):
    for tree in trees:
        print(tree)

for proj in projections1: print(proj)
for proj in projections2: print(proj)
for i, manifest in enumerate(single_tree_manifests):
    tree = manifest_to_tree(manifest)
    print(f"# Reconstructed tree {i}:")
    print(tree)
    print(f"## Projection on {''.join(restriction[0])}:")
    print(restrict_tree(tree, restriction[0]))
    print(f"## Projection on {''.join(restriction[1])}:")
    print(restrict_tree(tree, restriction[1]))

projection_ids1 = [proj.get_topology_id() for proj in projections1]
projection_ids2 = [proj.get_topology_id() for proj in projections2]
for i, manifest in enumerate(single_tree_manifests):
    tree = manifest_to_tree(manifest)
    print(f"# Reconstructed tree {i}:")
    print(tree)
    print(f"## Projection on {''.join(restriction[0])}:")
    restricted_tree1 = restrict_tree(tree, restriction[0])
    print(restricted_tree1)
    print(f"In restriction signature: {restricted_tree1.get_topology_id() in projection_ids1}")
    print(f"## Projection on {''.join(restriction[1])}:")
    restricted_tree2 = restrict_tree(tree, restriction[1])
    print(restricted_tree2)
    print(f"In restriction signature: {restricted_tree2.get_topology_id() in projection_ids2}")

# Experiment 13

X = "ABCD"
# all_trees_iter = generate_rooted(X)
X1 = "ABC"
X2 = "ABD"
restriction = (X1, X2)

result = visualize_restricted_rooted_trees_n(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

groupings1 = group_trees_by_restriction(generate_rooted(X), X1)
groupings2 = group_trees_by_restriction(generate_rooted(X), X2)
for id1, id2 in product(groupings1, groupings2):
    print(f"# Restriction signature: {id1, id2}")
    original_ids = sorted(tree.get_topology_id() for tree in result[(id1, id2)].treelist)
    print(f"Number of trees in original method: {len(original_ids)}")
    print(original_ids)
    new_ids = sorted(set(groupings1[id1]) & set(groupings2[id2]))
    print(f"Number of trees in new method: {len(new_ids)}")
    print(new_ids)
    print(original_ids == new_ids)

# Experiment 14

X = "ABCDE"
ambig = 1
n = 2
restrictions = full_cover_restrictions_n(X, ambig, n)
X1, X2 = restriction = next(restrictions)

result = visualize_restricted_rooted_trees_n(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

groupings1, id_to_tree1 = group_trees_by_restriction(generate_rooted(X), X1, keep_restricted_trees=True)
groupings2, id_to_tree2 = group_trees_by_restriction(generate_rooted(X), X2, keep_restricted_trees=True)
groupings1_keys = list(groupings1.keys())
groupings2_keys = list(groupings2.keys())

ids1 = [1, 5, 9]
ids2 = [4, 6, 12]

big_tree_ids1 = set().union(*(groupings1[groupings1_keys[i]] for i in ids1))
big_tree_ids2 = set().union(*(groupings2[groupings1_keys[i]] for i in ids2))
len(big_tree_ids1 & big_tree_ids2)

# TODO: get mutual trees via enumerating all trees and via manifest mutualization

# Experiment 14 (What is the smallest pair of trees, such that when you
# union their subsplits, you get virtual trees?)

X = "ABCDE"
tree_pairs = combinations(generate_rooted(X), 2)

found_one = False
tree_list_lens = []
for tree1, tree2 in tree_pairs:
    manifest1 = get_subsplit_manifest(tree1)
    manifest2 = get_subsplit_manifest(tree2)
    merged_manifest = merge_manifests((manifest1, manifest2))
    new_tree_list = possible_single_tree_manifests(merged_manifest)
    tree_list_lens.append(len(new_tree_list))
    if not found_one and len(new_tree_list) > 2:
        found_one = True
        print("Found one!")
        print(tree1)
        print(tree2)
max(tree_list_lens)
Counter(tree_list_lens)

