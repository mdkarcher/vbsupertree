import re
# import ete3

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


with open("data/BEAST/DS2.trees") as f:
    text_data = f.read()

# foo = text_data.split(" [&R] ")
foo = re.split(" \[\&R\] |\n", text_data)
foo2 = list(filter(lambda x: "[&rate=1.0]" in x, foo))
foo3 = list(map(lambda x: x.replace("[&rate=1.0]", ""), foo2))
foo4 = list(map(lambda x: MyTree(x), foo3))
foo_all = list(map(lambda x: MyTree(x), map(lambda x: x.replace("[&rate=1.0]", ""), filter(lambda x: "[&rate=1.0]" in x, re.split(" \[\&R\] |\n", text_data)))))
len(foo_all)
print(foo_all[0])
print(foo_all[100])

# DS2 experiments

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
# len(all_trees)
# len(all_trees[1001:])
trees = all_trees[5001:]
# trees[0].to_clade_set()

clade_counter = Counter()
for tree in trees:
    clade_counter.update(tree.to_clade_set())
# clade_counter
sorted(clade_counter.items(), key=lambda x: (-len(x[0]), -x[1]))

ranks = rank_focal_nodes(clade_counter)
ranks.most_common(10)

minus_focal_node, clade_focal_node = generate_restrictions_from_focal_node(focal_node='Homo_sapiens', clade_counter=clade_counter)

tree_counter = Counter(trees)
len(tree_counter)

subclade = Clade({'1', '11', '12', '13', '14', '15', '16', '18', '19', '2', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '4', '5', '7', '8', '9'})
# Clade({'1', '11', '12', '13', '14', '15', '19', '2', '22', '23', '24', '25', '26', '27', '28', '29', '3', '4', '5', '8', '9'})
# Clade({'1', '11', '12', '13', '14', '19', '22', '23', '25', '27', '28', '29', '4', '5', '8', '9'})
# Clade({'11', '25', '27', '29'})
focal_node = 11
# 11 Homo_sapiens

all_tips = trees[0].root_clade()
all_tips - subclade

# Using SBN restriction

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
trees = all_trees[1001:]

tree_dist = TreeDistribution.from_list(trees)
sbn = SCDSet.from_tree_distribution(tree_dist)

# restriction_minus11 = Clade({'1', '10', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '4', '5', '6', '7', '8', '9'})
# restriction_clade11 = Clade({'1', '11', '12', '13', '14', '15', '16', '18', '19', '2', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '4', '5', '7', '8', '9'})
restriction_minus, restriction_clade = generate_restrictions_from_focal_node(focal_node='Homo_sapiens', clade_counter=clade_counter)

sbn_minus = sbn.restrict(restriction_minus)
sbn_clade = sbn.restrict(restriction_clade)
support_minus = sbn_minus.support()
support_clade = sbn_clade.support()
mutual_support = support_minus.mutualize(support_clade)

starting_sbn = SCDSet.random_from_support(mutual_support)

supertree_sbn, kl_list, true_kl_list = scd_gradient_descent(
    starting=starting_sbn, references=[sbn_minus, sbn_clade],
    starting_gamma=2.0, max_iteration=200, true_reference=sbn
)

kl_list
true_kl_list
true_kl_list[100]

# Using separate BEAST runs

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
all_trees_minus = parse_beast_nexus("data/BEAST/DS2_minus11.trees")
all_trees_clade = parse_beast_nexus("data/BEAST/DS2_clade11.trees")
tree_dist = TreeDistribution.from_list(all_trees[1001:])
tree_dist_minus = TreeDistribution.from_list(all_trees_minus[1001:])
tree_dist_clade = TreeDistribution.from_list(all_trees_clade[1001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
sbn_minus = SCDSet.from_tree_distribution(tree_dist_minus)
sbn_clade = SCDSet.from_tree_distribution(tree_dist_clade)
support_minus = sbn_minus.support()
support_clade = sbn_clade.support()
mutual_support = support_minus.mutualize(support_clade)

mutual_support.is_complete()

starting_sbn = SCDSet.random_from_support(mutual_support)

# Currently raising a KeyError
supertree_sbn, kl_list, true_kl_list = scd_gradient_descent(
    starting=starting_sbn, references=[sbn_minus, sbn_clade],
    starting_gamma=2.0, max_iteration=200, true_reference=sbn
)

mutual_support_rest_minus = mutual_support.restrict(support_minus.root_clade())
mutual_support_rest_clade = mutual_support.restrict(support_clade.root_clade())

support_minus.to_set() - mutual_support_rest_minus.to_set()
mutual_support_rest_minus.to_set() - support_minus.to_set()

support_clade.to_set() - mutual_support_rest_clade.to_set()
mutual_support_rest_clade.to_set() - support_clade.to_set()

sbn_minus_pcss_probs = sbn_minus.pcss_probabilities()
sbn_clade_pcss_probs = sbn_clade.pcss_probabilities()

for pcss in support_minus.to_set() - mutual_support_rest_minus.to_set():
    print(f"{len(pcss.parent)}: {sbn_minus_pcss_probs.get(pcss):4.4g}")

len(support_minus.root_clade())

for pcss in support_clade.to_set() - mutual_support_rest_clade.to_set():
    print(f"{len(pcss.parent)}: {sbn_clade_pcss_probs.get(pcss):4.4g}")

len(support_clade.root_clade())

sum(sbn_minus_pcss_probs.get(pcss) for pcss in support_minus.to_set() - mutual_support_rest_minus.to_set())
sum(sbn_clade_pcss_probs.get(pcss) for pcss in support_clade.to_set() - mutual_support_rest_clade.to_set())


kl_list
true_kl_list
true_kl_list[100]

# Comparing restricted SBN to SBN of restricted BEAST run

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
all_trees_minus = parse_beast_nexus("data/BEAST/DS2_minus11.trees")
all_trees_clade = parse_beast_nexus("data/BEAST/DS2_clade11.trees")
tree_dist = TreeDistribution.from_list(all_trees[1001:])
tree_dist_minus = TreeDistribution.from_list(all_trees_minus[1001:])
tree_dist_clade = TreeDistribution.from_list(all_trees_clade[1001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
sbn_minus = SCDSet.from_tree_distribution(tree_dist_minus)
sbn_clade = SCDSet.from_tree_distribution(tree_dist_clade)

sbn_rest_minus = sbn.restrict(sbn_minus.root_clade())
sbn_rest_clade = sbn.restrict(sbn_clade.root_clade())

sbn_minus.kl_divergence(sbn_rest_minus)
sbn_clade.kl_divergence(sbn_rest_clade)

sbn_minus.support().to_set() - sbn_rest_minus.support().to_set()
sbn_rest_minus.support().to_set() - sbn_minus.support().to_set()

sbn_clade.support().to_set() - sbn_rest_clade.support().to_set()
sbn_rest_clade.support().to_set() - sbn_clade.support().to_set()

sbn_rest_minus.kl_divergence(sbn_minus)
sbn_rest_clade.kl_divergence(sbn_clade)

# Investigating bad key bug

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
all_trees_minus = parse_beast_nexus("data/BEAST/DS2_minus11.trees")
all_trees_clade = parse_beast_nexus("data/BEAST/DS2_clade11.trees")
tree_dist = TreeDistribution.from_list(all_trees[1001:])
tree_dist_minus = TreeDistribution.from_list(all_trees_minus[1001:])
tree_dist_clade = TreeDistribution.from_list(all_trees_clade[1001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
sbn_minus = SCDSet.from_tree_distribution(tree_dist_minus)
sbn_clade = SCDSet.from_tree_distribution(tree_dist_clade)
support_minus = sbn_minus.support()
support_clade = sbn_clade.support()
mutual_support = support_minus.mutualize(support_clade)

starting_sbn = SCDSet.random_from_support(mutual_support)
starting_sbn_rest_minus = starting_sbn.restrict(sbn_minus.root_clade())
starting_sbn_rest_clade = starting_sbn.restrict(sbn_clade.root_clade())

support_minus.is_complete()
support_clade.is_complete()
mutual_support.is_complete()
starting_sbn_rest_minus.support().is_complete()
starting_sbn_rest_clade.support().is_complete()

bad_ssc = SubsplitClade(Subsplit(Clade({'Acanthopleura_japonica', 'Artemia_salina', 'Eisenia_foetida', 'Eurypelma_californica', 'Lanice_conchilega', 'Limicolaria_kambeul', 'Placopecten_magellanicus', 'Priapulus_caudatus', 'Tenebrio_molitor'}), Clade({'Brachionus_plicatilis', 'Lepidodermella_squamata', 'Opisthorchis_viverrini'})), Clade({'Acanthopleura_japonica', 'Artemia_salina', 'Eisenia_foetida', 'Eurypelma_californica', 'Lanice_conchilega', 'Limicolaria_kambeul', 'Placopecten_magellanicus', 'Priapulus_caudatus', 'Tenebrio_molitor'}))
bad_ssc in support_minus.data
bad_ssc in starting_sbn_rest_minus.support().data
bad_ssc in support_clade.data
bad_ssc in starting_sbn_rest_clade.support().data
mutual_support_rest_minus = mutual_support.restrict(sbn_minus.root_clade())

# result, a PCSP in the support of one of the references does not show up in the support of the supertree SBN
# this leads to undefined KL divergence
# worry: perhaps the KL needs to be the other way around?
# Each PCSP in the supertree support must restrict to either a PCSP in the reference supports,
# or be at least partially trivial along a transit path that restricts to a PCSP in the reference supports.

# Flu parsing

# with open("data/BEAST/fluA.trees") as f:
#     text_data = f.read()

all_trees = parse_beast_nexus_test("data/BEAST/fluA.trees")
len(all_trees)
all_trees_set = set(all_trees)
len(all_trees_set)
print(all_trees[1001])
set(all_trees[1001].tree.get_leaf_names()) == set(all_trees[2001].tree.get_leaf_names())
all_trees[1001] == all_trees[2001]
trees = all_trees[1001:]

tree_dist = TreeDistribution.from_list(trees)
sbn = SCDSet.from_tree_distribution(tree_dist)
len(sbn)

clade_counter = Counter()
for tree in trees:
    clade_counter.update(tree.to_clade_set())
sorted_clades = sorted(clade_counter.items(), key=lambda x: (-len(x[0]), -x[1]))

ranks = rank_focal_nodes(clade_counter)
ranks.most_common(10)

focal_node = 'A_Waikato_12_1998'

restriction_minus, restriction_clade = generate_restrictions_from_focal_node(focal_node=focal_node, clade_counter=clade_counter)

sbn_minus = sbn.restrict(restriction_minus)
sbn_clade = sbn.restrict(restriction_clade)
support_minus = sbn_minus.support()
support_clade = sbn_clade.support()
mutual_support = support_minus.mutualize(support_clade)

starting_sbn = SCDSet.random_from_support(mutual_support)

supertree_sbn, kl_list, true_kl_list = scd_gradient_descent(
    starting=starting_sbn, references=[sbn_minus, sbn_clade],
    starting_gamma=2.0, max_iteration=200, true_reference=sbn
)

kl_list
true_kl_list
true_kl_list[100]

# ends up with all unique trees. Investigate more.

# Using separate BEAST runs, longer runs

all_trees_minus = parse_beast_nexus("data/BEAST/DS2_longer/DS2_minus11.trees")
all_trees_clade = parse_beast_nexus("data/BEAST/DS2_longer/DS2_clade11.trees")
tree_dist_minus = TreeDistribution.from_list(all_trees_minus[10001:])
tree_dist_clade = TreeDistribution.from_list(all_trees_clade[10001:])
sbn_minus = SCDSet.from_tree_distribution(tree_dist_minus)
sbn_clade = SCDSet.from_tree_distribution(tree_dist_clade)
support_minus = sbn_minus.support()
support_clade = sbn_clade.support()
mutual_support = support_minus.mutualize(support_clade)
mutual_support.is_complete()
mutual_support = mutual_support.prune()

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
tree_dist = TreeDistribution.from_list(all_trees[1001:])
sbn = SCDSet.from_tree_distribution(tree_dist)

starting_sbn = SCDSet.random_from_support(mutual_support)

# Currently raising a KeyError
supertree_sbn, kl_list, true_kl_list = scd_gradient_descent(
    starting=starting_sbn, references=[sbn_minus, sbn_clade],
    starting_gamma=2.0, max_iteration=200, #true_reference=sbn
)

mutual_support_rest_minus = mutual_support.restrict(support_minus.root_clade())
mutual_support_rest_clade = mutual_support.restrict(support_clade.root_clade())

support_minus.to_set() - mutual_support_rest_minus.to_set()
mutual_support_rest_minus.to_set() - support_minus.to_set()

support_clade.to_set() - mutual_support_rest_clade.to_set()
mutual_support_rest_clade.to_set() - support_clade.to_set()

sbn_minus_pcss_probs = sbn_minus.pcss_probabilities()
sbn_clade_pcss_probs = sbn_clade.pcss_probabilities()

for pcss in support_minus.to_set() - mutual_support_rest_minus.to_set():
    print(f"{sbn_minus_pcss_probs.get(pcss):4.4g}")

sum(sbn_minus_pcss_probs.get(pcss) for pcss in support_minus.to_set() - mutual_support_rest_minus.to_set())

sbn_minus_pcss_probs_sorted = sorted(sbn_minus_pcss_probs.items(), key=lambda x: -x[1])
sbn_minus_pcss_probs_sorted[:30]
sbn_minus_pcss_probs_sorted[-30:]

sum(sbn_clade_pcss_probs.get(pcss) for pcss in support_clade.to_set() - mutual_support_rest_clade.to_set())

# Comparing tree distributions of restricted single BEAST vs double BEAST

import gc

all_trees = parse_beast_nexus("data/BEAST/DS2_longer/DS2.trees")
len(all_trees)
tree_dist = TreeDistribution.from_list(all_trees[50001:])
del all_trees

all_trees_minus = parse_beast_nexus("data/BEAST/DS2_longer/DS2_minus11.trees")
len(all_trees_minus)
tree_dist_minus = TreeDistribution.from_list(all_trees_minus[50001:])
restriction_minus = all_trees_minus[-1].root_clade()
del all_trees_minus

all_trees_clade = parse_beast_nexus("data/BEAST/DS2_longer/DS2_clade11.trees")
len(all_trees_clade)
tree_dist_clade = TreeDistribution.from_list(all_trees_clade[50001:])
restriction_clade = all_trees_clade[-1].root_clade()
del all_trees_clade

gc.collect()

# restriction_minus = next(iter(tree_dist_minus.keys())).root_clade()
# restriction_clade = next(iter(tree_dist_clade.keys())).root_clade()
tree_dist_res_minus = tree_dist.restrict(restriction_minus)
tree_dist_res_clade = tree_dist.restrict(restriction_clade)
len(tree_dist)
len(tree_dist_minus)
len(tree_dist_clade)
len(tree_dist_res_minus)
len(tree_dist_res_clade)

foo = sorted(set(tree_dist_minus.keys()) | set(tree_dist_res_minus.keys()), key=lambda x: (-round(tree_dist_minus.get(x), 6), -round(tree_dist_res_minus.get(x), 6)))
for tree in foo:
    print(f"tree_dist_minus prob = {tree_dist_minus.get(tree):.6f}, tree_dist_res_minus prob = {tree_dist_res_minus.get(tree):.6f}")

foo = sorted(set(tree_dist_clade.keys()) | set(tree_dist_res_clade.keys()), key=lambda x: (-round(tree_dist_clade.get(x), 6), -round(tree_dist_res_clade.get(x), 6)))
for tree in foo:
    print(f"tree_dist_clade prob = {tree_dist_clade.get(tree):.6f}, tree_dist_res_clade prob = {tree_dist_res_clade.get(tree):.6f}")


## plus_minus experiments

all_trees = parse_beast_nexus("data/BEAST/DS2.trees")
all_trees_plushsapiens = parse_beast_nexus("data/BEAST/DS2_plusminus/DS2_plushsapiens_minuspmagellanicus.trees")
all_trees_pluspmagellanicus = parse_beast_nexus("data/BEAST/DS2_plusminus/DS2_pluspmagellancus_minushsapiens.trees")
tree_dist = TreeDistribution.from_list(all_trees[5001:])
tree_dist_plushsapiens = TreeDistribution.from_list(all_trees_plushsapiens[5001:])
tree_dist_pluspmagellanicus = TreeDistribution.from_list(all_trees_pluspmagellanicus[5001:])
sbn = SCDSet.from_tree_distribution(tree_dist)
sbn_plushsapiens = SCDSet.from_tree_distribution(tree_dist_plushsapiens)
sbn_pluspmagellanicus = SCDSet.from_tree_distribution(tree_dist_pluspmagellanicus)
support = sbn.support()
support_plushsapiens = sbn_plushsapiens.support()
support_pluspmagellanicus = sbn_pluspmagellanicus.support()
mutual_support = support_plushsapiens.mutualize(support_pluspmagellanicus)

mutual_support.is_complete()

in_fullsupport_notin_mutual = support.to_set() - mutual_support.to_set()
in_mutual_notin_fullsupport = mutual_support.to_set() - support.to_set()
len(in_fullsupport_notin_mutual), len(in_mutual_notin_fullsupport)

support_rest_plushsapiens = support.restrict(support_plushsapiens.root_clade())
support_rest_pluspmagellanicus = support.restrict(support_pluspmagellanicus.root_clade())
in_fullsupport_notin_plushsapiens = support_rest_plushsapiens.to_set() - support_plushsapiens.to_set()
in_fullsupport_notin_pluspmagellanicus = support_rest_pluspmagellanicus.to_set() - support_pluspmagellanicus.to_set()
in_plushsapiens_notin_fullsupport = support_plushsapiens.to_set() - support_rest_plushsapiens.to_set()
in_pluspmagellanicus_notin_fullsupport = support_pluspmagellanicus.to_set() - support_rest_pluspmagellanicus.to_set()
len(in_fullsupport_notin_plushsapiens), len(in_plushsapiens_notin_fullsupport)
len(in_fullsupport_notin_pluspmagellanicus), len(in_pluspmagellanicus_notin_fullsupport)

mutual_support_rest_plushsapiens = mutual_support.restrict(support_plushsapiens.root_clade())
mutual_support_rest_pluspmagellanicus = mutual_support.restrict(support_pluspmagellanicus.root_clade())

in_plushsapiens_notin_mutual = support_plushsapiens.to_set() - mutual_support_rest_plushsapiens.to_set()
in_mutual_notin_plushsapiens = mutual_support_rest_plushsapiens.to_set() - support_plushsapiens.to_set()
len(in_plushsapiens_notin_mutual), len(in_mutual_notin_plushsapiens)

in_pluspmagellanicus_notin_mutual = support_pluspmagellanicus.to_set() - mutual_support_rest_pluspmagellanicus.to_set()
in_mutual_notin_pluspmagellanicus = mutual_support_rest_pluspmagellanicus.to_set() - support_pluspmagellanicus.to_set()
len(in_pluspmagellanicus_notin_mutual), len(in_mutual_notin_pluspmagellanicus)

sbn_pcss_probs = sbn.pcss_probabilities()
sbn_plushsapiens_pcss_probs = sbn_plushsapiens.pcss_probabilities()
sbn_pluspmagellanicus_pcss_probs = sbn_pluspmagellanicus.pcss_probabilities()

for pcss in in_plushsapiens_notin_mutual:
    print(f"{len(pcss.parent)}: {sbn_plushsapiens_pcss_probs.get(pcss):4.4g}")

len(support_plushsapiens.root_clade())

for pcss in in_pluspmagellanicus_notin_mutual:
    print(f"{len(pcss.parent)}: {sbn_pluspmagellanicus_pcss_probs.get(pcss):4.4g}")

len(support_pluspmagellanicus.root_clade())

sum(sbn_plushsapiens_pcss_probs.get(pcss) for pcss in in_plushsapiens_notin_mutual)
sum(sbn_pluspmagellanicus_pcss_probs.get(pcss) for pcss in in_pluspmagellanicus_notin_mutual)

sbn_trim = sbn.copy()
sbn_trim.remove_many(in_fullsupport_notin_mutual)
sbn_trim._prune()
sbn_trim.normalize()
len(sbn_trim.support().to_set() - mutual_support.to_set())

sbn_plushsapiens_trim = sbn_plushsapiens.copy()
sbn_plushsapiens_trim.remove_many(in_plushsapiens_notin_mutual)
sbn_plushsapiens_trim._prune()
sbn_plushsapiens_trim.normalize()
len(sbn_plushsapiens_trim.support().to_set() - mutual_support_rest_plushsapiens.to_set())

sbn_pluspmagellanicus_trim = sbn_pluspmagellanicus.copy()
sbn_pluspmagellanicus_trim.remove_many(in_pluspmagellanicus_notin_mutual)
sbn_pluspmagellanicus_trim._prune()
sbn_pluspmagellanicus_trim.normalize()
len(sbn_pluspmagellanicus_trim.support().to_set() - mutual_support_rest_pluspmagellanicus.to_set())

starting_sbn = SCDSet.random_from_support(mutual_support)

supertree_sbn, kl_list, true_kl_list = scd_gradient_descent(
    starting=starting_sbn, references=[sbn_plushsapiens_trim, sbn_pluspmagellanicus_trim],
    starting_gamma=2.0, max_iteration=200, true_reference=sbn_trim
)

kl_list
true_kl_list
true_kl_list[100]

intersect = support_plushsapiens.root_clade() & support_pluspmagellanicus.root_clade()
sbn_plushsapiens_trim_rest = sbn_plushsapiens_trim.restrict(intersect)
sbn_pluspmagellanicus_trim_rest = sbn_pluspmagellanicus_trim.restrict(intersect)
sbn_plushsapiens_trim_rest.kl_divergence(sbn_pluspmagellanicus_trim_rest)
sbn_pluspmagellanicus_trim_rest.kl_divergence(sbn_plushsapiens_trim_rest)
sbn_plushsapiens_trim_rest.support().to_set() - sbn_pluspmagellanicus_trim_rest.support().to_set()
sbn_pluspmagellanicus_trim_rest.support().to_set() - sbn_plushsapiens_trim_rest.support().to_set()

tree_dist_plushsapiens_rest = tree_dist_plushsapiens.restrict(intersect)
tree_dist_pluspmagellanicus_rest = tree_dist_pluspmagellanicus.restrict(intersect)
tree_dist_plushsapiens_rest.kl_divergence(tree_dist_pluspmagellanicus_rest)
tree_dist_pluspmagellanicus_rest.kl_divergence(tree_dist_plushsapiens_rest)
len(set(tree_dist_plushsapiens_rest.keys()) - set(tree_dist_pluspmagellanicus_rest.keys()))
len(set(tree_dist_pluspmagellanicus_rest.keys()) - set(tree_dist_plushsapiens_rest.keys()))
len(tree_dist_plushsapiens_rest.keys())
len(tree_dist_pluspmagellanicus_rest.keys())

tree_dist_rest_plushsapiens = tree_dist.restrict(support_plushsapiens.root_clade())
tree_dist_rest_pluspmagellanicus = tree_dist.restrict(support_pluspmagellanicus.root_clade())
len(set(tree_dist_rest_plushsapiens.keys()) - set(tree_dist_plushsapiens.keys())), len(tree_dist_rest_plushsapiens), len(tree_dist_plushsapiens), len(set(tree_dist_plushsapiens.keys()) - set(tree_dist_rest_plushsapiens.keys()))
len(set(tree_dist_rest_pluspmagellanicus.keys()) - set(tree_dist_pluspmagellanicus.keys())), len(tree_dist_rest_pluspmagellanicus), len(tree_dist_pluspmagellanicus), len(set(tree_dist_pluspmagellanicus.keys()) - set(tree_dist_rest_pluspmagellanicus.keys()))

tree_probs_plushsapiens = [(tree, tree_dist_rest_plushsapiens[tree], tree_dist_plushsapiens[tree]) for tree in set(tree_dist_rest_plushsapiens.keys()) | set(tree_dist_plushsapiens.keys())]
tree_probs_pluspmagellanicus = [(tree, tree_dist_rest_pluspmagellanicus[tree], tree_dist_pluspmagellanicus[tree]) for tree in set(tree_dist_rest_pluspmagellanicus.keys()) | set(tree_dist_pluspmagellanicus.keys())]

sorted_tree_probs_plushsapiens = sorted(tree_probs_plushsapiens, key=lambda x: (-x[1], -x[2]))
for i, (tree, rest_plus, plus) in enumerate(sorted_tree_probs_plushsapiens):
    print(f"tree {i}: \t rest_plus = {rest_plus:.6f} \t plus = {plus:.6f}")

sorted_tree_probs_pluspmagellanicus = sorted(tree_probs_pluspmagellanicus, key=lambda x: (-x[1], -x[2]))
for i, (tree, rest_plus, plus) in enumerate(sorted_tree_probs_pluspmagellanicus):
    print(f"tree {i}: \t rest_plus = {rest_plus:.6f} \t plus = {plus:.6f}")

# re-run comparisons

all_trees1 = parse_beast_nexus("data/BEAST/DS2.trees")
all_trees2 = parse_beast_nexus("data/BEAST/DS2_run02/DS2.trees")
all_trees_plushsapiens1 = parse_beast_nexus("data/BEAST/DS2_plusminus/DS2_plushsapiens_minuspmagellanicus.trees")
all_trees_plushsapiens2 = parse_beast_nexus("data/BEAST/DS2_run02/DS2_plushsapiens_minuspmagellanicus.trees")
all_trees_pluspmagellanicus1 = parse_beast_nexus("data/BEAST/DS2_plusminus/DS2_pluspmagellancus_minushsapiens.trees")
all_trees_pluspmagellanicus2 = parse_beast_nexus("data/BEAST/DS2_run02/DS2_pluspmagellancus_minushsapiens.trees")
tree_dist1 = TreeDistribution.from_list(all_trees1[5001:])
tree_dist2 = TreeDistribution.from_list(all_trees2[5001:])
tree_dist_plushsapiens1 = TreeDistribution.from_list(all_trees_plushsapiens1[5001:])
tree_dist_plushsapiens2 = TreeDistribution.from_list(all_trees_plushsapiens2[5001:])
tree_dist_pluspmagellanicus1 = TreeDistribution.from_list(all_trees_pluspmagellanicus1[5001:])
tree_dist_pluspmagellanicus2 = TreeDistribution.from_list(all_trees_pluspmagellanicus2[5001:])
sbn1 = SCDSet.from_tree_distribution(tree_dist1)
sbn2 = SCDSet.from_tree_distribution(tree_dist2)
sbn_plushsapiens1 = SCDSet.from_tree_distribution(tree_dist_plushsapiens1)
sbn_plushsapiens2 = SCDSet.from_tree_distribution(tree_dist_plushsapiens2)
sbn_pluspmagellanicus1 = SCDSet.from_tree_distribution(tree_dist_pluspmagellanicus1)
sbn_pluspmagellanicus2 = SCDSet.from_tree_distribution(tree_dist_pluspmagellanicus2)
support1 = sbn1.support()
support2 = sbn2.support()
support_plushsapiens1 = sbn_plushsapiens1.support()
support_plushsapiens2 = sbn_plushsapiens2.support()
support_pluspmagellanicus1 = sbn_pluspmagellanicus1.support()
support_pluspmagellanicus2 = sbn_pluspmagellanicus2.support()

tree_probs = sorted([(tree, tree_dist1[tree], tree_dist2[tree]) for tree in set(tree_dist1.keys()) | set(tree_dist2.keys())], key=lambda x: (-x[1], -x[2]))
tree_probs_plushsapiens = sorted([(tree, tree_dist_plushsapiens1[tree], tree_dist_plushsapiens2[tree]) for tree in set(tree_dist_plushsapiens1.keys()) | set(tree_dist_plushsapiens2.keys())], key=lambda x: (-x[1], -x[2]))
tree_probs_pluspmagellanicus = sorted([(tree, tree_dist_pluspmagellanicus1[tree], tree_dist_pluspmagellanicus2[tree]) for tree in set(tree_dist_pluspmagellanicus1.keys()) | set(tree_dist_pluspmagellanicus2.keys())], key=lambda x: (-x[1], -x[2]))

for i, (tree, prob1, prob2) in enumerate(tree_probs):
    print(f"tree {i}: \t prob1 = {prob1:.6f} \t prob2 = {prob2:.6f}")

for i, (tree, prob1, prob2) in enumerate(tree_probs_plushsapiens):
    print(f"tree {i}: \t prob1 = {prob1:.6f} \t prob2 = {prob2:.6f}")

for i, (tree, prob1, prob2) in enumerate(tree_probs_pluspmagellanicus):
    print(f"tree {i}: \t prob1 = {prob1:.6f} \t prob2 = {prob2:.6f}")



