import re
import numpy as np
import pandas as pd
import ete3
from collections import Counter
from dendropy_code import *

from importlib import reload
import classes

reload(classes)
from classes import *


def parse_translation(translate_text):
    item_list = re.findall("[0-9]+ \w+", translate_text)
    result = dict()
    for item in item_list:
        fr, to = item.split(" ")
        result[fr] = to
    return result


def parse_beast_nexus(filename):
    with open(filename) as f:
        text_data = f.read()
    text_sections = text_data.split(";\n")
    translate_text = next(section for section in text_sections if "Translate" in section)
    map_dict = parse_translation(translate_text)

    tree_sections = filter(lambda x: "tree " in x, text_sections)
    trees = []
    for tree_section in tree_sections:
        _, newick_section = tree_section.split(" [&R] ")
        newick_fixed = newick_section.replace("[&rate=1.0]", "")
        tree = MyTree(newick_fixed+";")
        tree.rename_tips(map_dict)
        trees.append(tree)
    return trees


def parse_beast_nexus_test(filename):
    with open(filename) as f:
        text_data = f.read()
    text_sections = text_data.split(";\n")
    translate_text = next(section for section in text_sections if "Translate" in section)
    map_dict = parse_translation(translate_text)

    tree_sections = filter(lambda x: "tree " in x, text_sections)
    trees = []
    for tree_section in tree_sections:
        _, newick_section = tree_section.split(" [&R] ")
        # newick_fixed = re.sub(r"\[.*\]", "", newick_section)
        newick_fixed = re.sub(r"\[&rate=\d+.\d+(e|E)?-?\d*\]", "", newick_section)
        # newick_fixed = newick_section.replace("[&rate=1.0]", "")
        tree = MyTree(newick_fixed+";")
        tree.rename_tips(map_dict)
        trees.append(tree)
    return trees


def rank_focal_nodes(clade_counter):
    max_n = clade_counter.most_common(1)[0][1]
    scores = Counter()
    for clade, n in clade_counter.items():
        if n == max_n:
            k = len(clade)
            for tip in clade:
                scores[tip] += k
    return scores


def generate_restrictions_from_focal_node(focal_node, clade_counter):
    root_clade = max(clade_counter.items(), key=lambda x: len(x[0]))[0]
    max_n = clade_counter[root_clade]
    minus_focal_node = root_clade - Clade({focal_node})
    candidates = filter(lambda x: x[1] == max_n and x[0] != root_clade and focal_node in x[0], clade_counter.items())
    clade_focal_node = max(candidates, key=lambda x: len(x[0]))[0]
    return minus_focal_node, clade_focal_node


# Loading all_tips.trees

df = pd.read_table("data/all_tips.trees", )
trees = df['tree'].tolist()

MyTree(trees[0], format=1)
ete3.Tree(trees[2], format=1).get_leaf_names()

tipset = set()
for tree_str in trees:
    tipset |= set(MyTree(tree_str, format=1).tree.get_leaf_names())
tipset

count = Counter()
for tree_str in trees:
    mytree = MyTree(tree_str, format=1)
    count[mytree] += 1
count
len(count)

for tree in count:
    print(tree)

tree_dist = TreeDistribution.from_list([MyTree(tree_str, format=1) for tree_str in trees])
print(tree_dist)

# Making and restricting SBNs

import numpy as np
import pandas as pd
import random
from importlib import reload
import classes

reload(classes)
from classes import *


def inner_flow(true_scd, references, support):
    starting_scd = SCDSet.random_from_support(support)
    scd, kl_list, true_kl_list = scd_gradient_descent(
        starting=starting_scd, references=references,
        starting_gamma=2.0, max_iteration=100, true_reference=true_scd
    )
    return kl_list, true_kl_list


def inner_loop(n, true_scd, references, support):
    kl_list_of_lists = []
    true_kl_list_of_lists = []
    for _ in range(n):
        kl_list, true_kl_list = inner_flow(true_scd, references, support)
        kl_list_of_lists.append(kl_list)
        true_kl_list_of_lists.append(true_kl_list)
    return np.array(kl_list_of_lists), np.array(true_kl_list_of_lists)


def outer_flow(n, true_scd, references):
    support = references[0].support()
    for i in range(1, len(references)):
        support = support.mutualize(references[i].support())
    kl_array, true_kl_array = inner_loop(n, true_scd, references, support)
    return kl_array, true_kl_array


def outer_loop(n, true_scd, all_references):
    kl_list_of_arrays = []
    true_kl_list_of_arrays = []
    for k in range(2, len(all_references)+1):
        references = random.sample(all_references, k)
        kl_array, true_kl_array = outer_flow(n, true_scd, references)
        kl_list_of_arrays.append(kl_array)
        true_kl_list_of_arrays.append(true_kl_array)
    kl_3d_array = np.array(kl_list_of_arrays)
    true_kl_3d_array = np.array(true_kl_list_of_arrays)
    return kl_3d_array, true_kl_3d_array


## loop experiment

X = Clade("ABCDE")
restrictions = [Clade("ABCD"), Clade("ABCE"), Clade("ABDE"), Clade("ACDE"), Clade("BCDE")]

true_scd = SCDSet.random(X)
all_references = [true_scd.restrict(restriction) for restriction in restrictions]

kl_array, true_kl_array = outer_flow(2, true_scd, all_references)

kl_3d_array, true_kl_3d_array = outer_loop(2, true_scd, all_references)

## DS1 toy experiment

df = pd.read_table("data/all_tips.trees")
trees = df['tree'].tolist()
tree_dist = TreeDistribution.from_list([MyTree(tree_str, format=1) for tree_str in trees])

true_sbn = SCDSet.from_tree_distribution(tree_dist)
tips = list(next(iter(tree_dist.keys())).tree.get_leaf_names())

restrictions = [tips[:idx] + tips[idx+1:] for idx in range(len(tips))]
all_references = [true_sbn.restrict(restriction) for restriction in restrictions]

n = 21
ds1_kl_3d_array, ds1_true_kl_3d_array = outer_loop(n, true_sbn, all_references)
np.save("objects/ds1_kl_3d_array.npy", ds1_kl_3d_array)
np.save("objects/ds1_true_kl_3d_array.npy", ds1_true_kl_3d_array)

ds1_kl_medians = np.median(ds1_kl_3d_array, 1)
ds1_true_kl_medians = np.median(ds1_true_kl_3d_array, 1)

ds1_kl_95perc = np.percentile(ds1_kl_3d_array, 95, 1)
ds1_true_kl_95perc = np.percentile(ds1_true_kl_3d_array, 95, 1)

ds1_kl_05perc = np.percentile(ds1_kl_3d_array, 5, 1)
ds1_true_kl_05perc = np.percentile(ds1_true_kl_3d_array, 5, 1)

np.max(ds1_kl_3d_array)
np.max(ds1_true_kl_3d_array)
np.min(ds1_kl_3d_array)
np.min(ds1_true_kl_3d_array)

from matplotlib import pyplot as plt


def plot_kls(kl_array, true_kl_array):
    k, n, it = kl_array.shape
    full_min = min(np.min(kl_array), np.min(true_kl_array))
    full_max = max(np.max(kl_array), np.max(true_kl_array))
    kl_med = np.median(kl_array, 1)
    kl_95p = np.percentile(kl_array, 95, 1)
    kl_05p = np.percentile(kl_array,  5, 1)
    true_kl_med = np.median(true_kl_array, 1)
    true_kl_95p = np.percentile(true_kl_array, 95, 1)
    true_kl_05p = np.percentile(true_kl_array,  5, 1)
    fig, axs = plt.subplots(2, k, sharex=True, sharey=True)
    plt.yscale('log')
    plt.ylim(full_min, full_max)
    x = list(range(it))
    for i in range(k):
        ax_kl = axs[0, i]
        ax_kl.set_title(f"{i+2} References")
        ax_kl.set(xlabel="Iteration", ylabel="Loss Function")
        ax_kl.label_outer()
        ax_kl.plot(x, kl_med[i, :])
        ax_kl.fill_between(x, kl_05p[i, :], kl_95p[i, :], color='b', alpha=0.1)
        ax_true_kl = axs[1, i]
        ax_true_kl.set(xlabel="Iteration", ylabel="KL vs Truth")
        ax_true_kl.label_outer()
        ax_true_kl.plot(x, true_kl_med[i, :])
        ax_true_kl.fill_between(x, true_kl_05p[i, :], true_kl_95p[i, :], color='b', alpha=0.1)
        # if i == 0:
        #     ax_kl.set_title('Loss Function')
        #     ax_true_kl.set_title('KL vs Truth')
    return fig, axs


x = list(range(101))
top = ds1_kl_95perc[0, :]
mid = ds1_kl_medians[0, :]
bot = ds1_kl_05perc[0, :]

fig, ax = plt.subplots()
plt.yscale('log')
ax.plot(x, mid)
ax.fill_between(x, bot, top, color='b', alpha=.1)


plot_kls(ds1_kl_3d_array, ds1_true_kl_3d_array)

# Erick simulation

import dendropy as dp

true_tree_str = "(outgroup:0.5,(((z0:0.1,z8:0.1):0.1,(z4:0.1,z7:0.1):0.1):0.1,(z5:0.1,((z1:0.1,(z3:0.1,z6:0.1):0.1):0.1,z2:0.1):0.1):0.1):0.1);"
true_tree_dp = dp.Tree.get(data=true_tree_str, schema="newick")
true_tree_dp.print_plot()
true_tree_dp.calc_node_root_distances()
for node in true_tree_dp.leaf_node_iter():
    print(f"{node.taxon}: {node.root_distance:0.4g}")

true_tree = MyTree(tree="data/Erick_sims/run.nwk")
print(true_tree)

all_trees = parse_beast_nexus_test("data/Erick_sims/run01.trees")
tree_dist = TreeDistribution.from_list(all_trees[5001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
support = sbn.support()

tree_dist[true_tree]


# dendropy simulation experiments

simulate(taxon_count=20, seq_len=200, prefix="data/dendropy_sims/tc20_sl200_01")
tree = MyTree("data/dendropy_sims/tc20_sl200_01.nwk")
print(tree)
tree.tree.show()

dp_tree = dp.Tree.get(path="data/dendropy_sims/tc20_sl200_01.nwk", schema="newick")
dp_tree.print_plot()
dp_tree.calc_node_root_distances()

# Run BEAUTi/BEAST here

true_tree = MyTree("data/dendropy_sims/tc20_sl200_01.nwk")
all_trees = parse_beast_nexus_test("data/dendropy_sims/tc20_sl200_01_01.trees")
tree_dist = TreeDistribution.from_list(all_trees[5001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
support = sbn.support()

tree_dist[true_tree]
len(tree_dist)
sorted_tree_dist = sorted(tree_dist.items(), key=lambda x: -x[1])
for i, (tree, prob) in enumerate(sorted_tree_dist):
    print(f"{i}: {prob:.5g}")
    if tree == true_tree:
        print("Note: matches true tree.")

print(sorted_tree_dist[0][0])
print(true_tree)

# tc20_sl200_01 experiments

true_tree = MyTree("data/dendropy_sims/tc20_sl200_01.nwk")
all_trees = parse_beast_nexus("data/dendropy_sims/tc20_sl200_01_01.trees")
all_trees_minus_z9 = parse_beast_nexus("data/dendropy_sims/tc20_sl200_01_minus_z9_01.trees")
all_trees_minus_z10 = parse_beast_nexus("data/dendropy_sims/tc20_sl200_01_minus_z10_01.trees")
all_trees_minus_z9z10 = parse_beast_nexus("data/dendropy_sims/tc20_sl200_01_minus_z9z10_01.trees")
tree_dist = TreeDistribution.from_list(all_trees[5001:])
tree_dist_minus_z9 = TreeDistribution.from_list(all_trees_minus_z9[5001:])
tree_dist_minus_z10 = TreeDistribution.from_list(all_trees_minus_z10[5001:])
tree_dist_minus_z9z10 = TreeDistribution.from_list(all_trees_minus_z9z10[5001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
sbn_minus_z9 = SCDSet.from_tree_distribution(tree_dist_minus_z9)
sbn_minus_z10 = SCDSet.from_tree_distribution(tree_dist_minus_z10)
sbn_minus_z9z10 = SCDSet.from_tree_distribution(tree_dist_minus_z9z10)
support = sbn.support()
support_minus_z9 = sbn_minus_z9.support()
support_minus_z10 = sbn_minus_z10.support()
support_minus_z9z10 = sbn_minus_z9z10.support()
mutual_support = support_minus_z9.mutualize(support_minus_z10)

len(mutual_support)
mutual_support.is_complete()
mutual_support = mutual_support.prune()

tree_dist_rest_minus_z9 = tree_dist.restrict(support_minus_z9.root_clade())
tree_dist_rest_minus_z10 = tree_dist.restrict(support_minus_z10.root_clade())
true_tree_rest_minus_z9 = true_tree.restrict(support_minus_z9.root_clade())
true_tree_rest_minus_z10 = true_tree.restrict(support_minus_z10.root_clade())
len(set(tree_dist_rest_minus_z9.keys()) - set(tree_dist_minus_z9.keys())), len(tree_dist_rest_minus_z9), len(tree_dist_minus_z9), len(set(tree_dist_minus_z9.keys()) - set(tree_dist_rest_minus_z9.keys()))
len(set(tree_dist_rest_minus_z10.keys()) - set(tree_dist_minus_z10.keys())), len(tree_dist_rest_minus_z10), len(tree_dist_minus_z10), len(set(tree_dist_minus_z10.keys()) - set(tree_dist_rest_minus_z10.keys()))

tree_probs_minus_z9 = [(tree, tree_dist_rest_minus_z9[tree], tree_dist_minus_z9[tree]) for tree in set(tree_dist_rest_minus_z9.keys()) | set(tree_dist_minus_z9.keys())]
tree_probs_minus_z10 = [(tree, tree_dist_rest_minus_z10[tree], tree_dist_minus_z10[tree]) for tree in set(tree_dist_rest_minus_z10.keys()) | set(tree_dist_minus_z10.keys())]

sorted_tree_probs_minus_z9 = sorted(tree_probs_minus_z9, key=lambda x: (-x[1], -x[2]))
for i, (tree, rest_plus, plus) in enumerate(sorted_tree_probs_minus_z9):
    print(f"tree {i}: \t rest_plus = {rest_plus:.6f} \t plus = {plus:.6f}")
    if tree == true_tree_rest_minus_z9:
        print("Matches true tree")

sorted_tree_probs_minus_z10 = sorted(tree_probs_minus_z10, key=lambda x: (-x[1], -x[2]))
for i, (tree, rest_plus, plus) in enumerate(sorted_tree_probs_minus_z10):
    print(f"tree {i}: \t rest_plus = {rest_plus:.6f} \t plus = {plus:.6f}")
    if tree == true_tree_rest_minus_z10:
        print("Matches true tree")

tree_dist_dust = tree_dist.dust(0.002)
tree_dist_minus_z9_dust = tree_dist_minus_z9.dust(0.002)
tree_dist_minus_z10_dust = tree_dist_minus_z10.dust(0.002)
tree_dist_minus_z9z10_dust = tree_dist_minus_z9z10.dust(0.002)

sbn_dust = SCDSet.from_tree_distribution(tree_dist_dust)
sbn_minus_z9_dust = SCDSet.from_tree_distribution(tree_dist_minus_z9_dust)
sbn_minus_z10_dust = SCDSet.from_tree_distribution(tree_dist_minus_z10_dust)
sbn_minus_z9z10_dust = SCDSet.from_tree_distribution(tree_dist_minus_z9z10_dust)
support_dust = sbn_dust.support()
support_minus_z9_dust = sbn_minus_z9_dust.support()
support_minus_z10_dust = sbn_minus_z10_dust.support()
support_minus_z9z10_dust = sbn_minus_z9z10_dust.support()
mutual_support_dust = support_minus_z9_dust.mutualize(support_minus_z10_dust)
len(mutual_support_dust)
mutual_support_dust.is_complete()
mutual_support_dust = mutual_support_dust.prune()

in_support_dust_notin_mutual_dust = support_dust.to_set() - mutual_support_dust.to_set()
in_mutual_dust_notin_support_dust = mutual_support_dust.to_set() - support_dust.to_set()
len(in_support_dust_notin_mutual_dust), len(support_dust), len(mutual_support_dust), len(in_mutual_dust_notin_support_dust)
# (14, 141, 135, 8)

mutual_support_dust_rest_minus_z9 = mutual_support_dust.restrict(support_minus_z9_dust.root_clade())
mutual_support_dust_rest_minus_z10 = mutual_support_dust.restrict(support_minus_z10_dust.root_clade())

in_minus_z9_dust_notin_mutual_dust = support_minus_z9_dust.to_set() - mutual_support_dust_rest_minus_z9.to_set()
in_mutual_dust_notin_minus_z9_dust = mutual_support_dust_rest_minus_z9.to_set() - support_minus_z9_dust.to_set()
len(in_minus_z9_dust_notin_mutual_dust), len(support_minus_z9_dust), len(mutual_support_dust_rest_minus_z9), len(in_mutual_dust_notin_minus_z9_dust)
# (3, 137, 134, 0)

in_minus_z10_dust_notin_mutual_dust = support_minus_z10_dust.to_set() - mutual_support_dust_rest_minus_z10.to_set()
in_mutual_dust_notin_minus_z10_dust = mutual_support_dust_rest_minus_z10.to_set() - support_minus_z10_dust.to_set()
len(in_minus_z10_dust_notin_mutual_dust), len(support_minus_z10_dust), len(mutual_support_dust_rest_minus_z10), len(in_mutual_dust_notin_minus_z10_dust)
# (21, 146, 125, 0)

sbn_dust_trim = sbn_dust.copy()
sbn_dust_trim.remove_many(in_support_dust_notin_mutual_dust)
sbn_dust_trim._prune()
sbn_dust_trim.normalize()
len(sbn_dust_trim)
len(sbn_dust)

sbn_minus_z9_dust_trim = sbn_minus_z9_dust.copy()
sbn_minus_z9_dust_trim.remove_many(in_minus_z9_dust_notin_mutual_dust)
sbn_minus_z9_dust_trim._prune()
sbn_minus_z9_dust_trim.normalize()
len(sbn_minus_z9_dust_trim)
len(sbn_minus_z9_dust)

sbn_minus_z10_dust_trim = sbn_minus_z10_dust.copy()
sbn_minus_z10_dust_trim.remove_many(in_minus_z10_dust_notin_mutual_dust)
sbn_minus_z10_dust_trim._prune()
sbn_minus_z10_dust_trim.normalize()
len(sbn_minus_z10_dust_trim)
len(sbn_minus_z10_dust)

support_dust_trim = sbn_dust_trim.support()
support_minus_z9_dust_trim = sbn_minus_z9_dust_trim.support()
support_minus_z10_dust_trim = sbn_minus_z10_dust_trim.support()
mutual_support_minus_z9_dust_trim_minus_z10_dust_trim = support_minus_z9_dust_trim.mutualize(support_minus_z10_dust_trim)
len(mutual_support_minus_z9_dust_trim_minus_z10_dust_trim)
mutual_support_minus_z9_dust_trim_minus_z10_dust_trim.is_complete()
mutual_support_minus_z9_dust_trim_minus_z10_dust_trim = mutual_support_minus_z9_dust_trim_minus_z10_dust_trim.prune()

len(support_dust_trim.to_set() - mutual_support_minus_z9_dust_trim_minus_z10_dust_trim.to_set())
len(support_minus_z9_dust_trim.to_set() - mutual_support_minus_z9_dust_trim_minus_z10_dust_trim.restrict(support_minus_z9_dust_trim.root_clade()).to_set())
len(support_minus_z10_dust_trim.to_set() - mutual_support_minus_z9_dust_trim_minus_z10_dust_trim.restrict(support_minus_z10_dust_trim.root_clade()).to_set())

starting_sbn = SCDSet.random_from_support(mutual_support_minus_z9_dust_trim_minus_z10_dust_trim)

sbn_minus_z9_dust_trim.kl_divergence(starting_sbn.restrict(sbn_minus_z9_dust_trim.root_clade()))
sbn_minus_z10_dust_trim.kl_divergence(starting_sbn.restrict(sbn_minus_z10_dust_trim.root_clade()))
sbn_dust_trim.kl_divergence(starting_sbn)
len(sbn_dust_trim.support().to_set() - starting_sbn.support().to_set()), len(sbn_dust_trim.support().to_set()), len(starting_sbn.support().to_set())

# investigating error
sbn_dust_trim_pcsss = sbn_dust_trim.pcss_probabilities(verbose=True)

supertree_sbn, kl_list, true_kl_list = scd_gradient_descent(
    starting=starting_sbn, references=[sbn_minus_z9_dust_trim, sbn_minus_z10_dust_trim],
    starting_gamma=2.0, max_iteration=50, true_reference=sbn_dust_trim
)

kl_list
true_kl_list
true_kl_list[20]
true_kl_list[-1]
x = range(len(kl_list))

from matplotlib import pyplot as plt
import seaborn as sns

# Horizontal
sns.set_context("poster")
fig, (ax_kl, ax_true_kl) = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, sharex=False)
# fig.suptitle("vbsupertree convergence")
ax_kl.set(title="Loss function", xlabel="Iteration", ylabel="Nats", yscale="log")
# ax_kl.label_outer()
ax_kl.plot(kl_list)
ax_true_kl.set(title="KL vs. Truth", xlabel="Iteration", ylabel="", yscale="log")
# ax_true_kl.label_outer()
ax_true_kl.plot(true_kl_list)
plt.tight_layout()
plt.savefig("figures/vbsupertree.pdf", format="pdf")

# Vertical
sns.set_context("poster")
fig, (ax_kl, ax_true_kl) = plt.subplots(2, 1, figsize=(6, 10), sharey=False, sharex=False)
# fig.suptitle("vbsupertree convergence")
ax_kl.set(title="", xlabel="", ylabel="Loss function", yscale="log")
# ax_kl.label_outer()
ax_kl.plot(kl_list)
ax_true_kl.set(title="", xlabel="Iteration", ylabel="KL vs. Truth", yscale="log")
# ax_true_kl.label_outer()
ax_true_kl.plot(true_kl_list)
plt.tight_layout()
plt.savefig("figures/vbsupertree_vert.pdf", format="pdf")

# formatting some trees for Erick

from vbsupertree import *

all_trees = parse_beast_nexus("data/dendropy_sims/tc20_sl200_01_01.trees")
trees_subset = all_trees[-100:]
tree_strs = [tree.tree.write(format=9)+'\n' for tree in trees_subset]
with open("data/dendropy_sims/Erick_tc20_sl200.nwk", 'w') as f:
    f.writelines(tree_strs)

tree_dist = TreeDistribution.from_list(trees_subset)
sbn = SBN.from_tree_distribution(tree_dist)
bit_map = sorted(sbn.root_clade())
bit_summary = sbn.bitarray_summary(bit_map)

csv_strs = [f"{bit_str}, {value}\n" for bit_str, value in bit_summary.items()]
with open("data/dendropy_sims/Erick_tc20_sl200_sbn.csv", 'w') as f:
    f.writelines(csv_strs)


