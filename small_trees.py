from ete3 import Tree
# from bitarray import bitarray
from collections import Counter
from collections import namedtuple
from collections import defaultdict
from itertools import combinations
from itertools import product
import copy
import random


ProjectionTrees = namedtuple('ProjectionTrees', ['proj1', 'proj2', 'treelist'])
ProjectionTrees_n = namedtuple('ProjectionTrees_n', ['projections', 'treelist'])
TreeRecursion = namedtuple("TreeRecursion", ['subsplit_set', 'clade_stack'])


def generate_rooted(taxa):
    if len(taxa) == 1:
        yield Tree(f"{taxa[0]};")
    elif len(taxa) == 2:
        yield Tree(f"({','.join(taxa)});")
    else:
        for tree in generate_unrooted(taxa):
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    tree.set_outgroup(node)
                    yield tree.copy()


def generate_unrooted(taxa):
    if len(taxa) <= 3:
        # return [Tree('(' + ','.join(taxa) + ');')]
        yield Tree('(' + ','.join(taxa) + ');')
    else:
        # res = []
        sister = Tree('(' + taxa[-1] + ');')
        for tree in generate_unrooted(taxa[:-1]):
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    node.up.add_child(sister)
                    node.detach()
                    sister.add_child(node)

                    # res.append(copy.deepcopy(tree))
                    # res.append(tree.copy())
                    yield tree.copy()

                    node.detach()
                    sister.up.add_child(node)
                    sister.detach()
        # return res


def get_leaf_set(tree):
    return {leaf.name for leaf in tree.get_leaves()}


def restrict_tree(tree, taxa):
    restriction = get_leaf_set(tree) & set(taxa)
    if not restriction:
        return None
    if len(restriction) == 1:
        return Tree(f"{''.join(restriction)};")
    tree_cp = tree.copy()
    tree_cp.prune(restriction)
    return tree_cp


def explore_rooted_trees(taxa, restrictions, full_union_only=True):
    trees_iter = generate_rooted(taxa)
    restrictions_combinations = [(tup1, tup2) for tup1, tup2 in combinations(restrictions, 2) if not full_union_only or len(set(tup1) | set(tup2)) == len(taxa)]
    counters = {}
    for comb in restrictions_combinations:
        counters[comb] = Counter()
    for tree in trees_iter:
        for comb in restrictions_combinations:
            projection1 = restrict_tree(tree, comb[0])
            projection2 = restrict_tree(tree, comb[1])
            counters[comb][projection1.get_topology_id(), projection2.get_topology_id()] += 1
    return counters


def visualize_restricted_rooted_trees(taxa, restriction):
    result = {}
    for tree in generate_rooted(taxa):
        projection1 = restrict_tree(tree, restriction[0])
        projection2 = restrict_tree(tree, restriction[1])
        signature = (projection1.get_topology_id(), projection2.get_topology_id())
        if signature not in result:
            result[signature] = ProjectionTrees(projection1, projection2, [])
        result[signature].treelist.append(tree)
    return result


def subsplit_to_string(subsplit):
    if len(subsplit) < 2:
        return ":"
    assert len(subsplit) == 2
    left, right = subsplit
    left_str = ''.join(sorted(left))
    right_str = ''.join(sorted(right))
    str_list = sorted((left_str, right_str))
    return ':'.join(str_list)


def string_to_subsplit(subsplit_str):
    left, right = subsplit_str.split(':')
    return frozenset({frozenset(left), frozenset(right)})


def get_subsplits(tree, restriction=None, to_string=True):
    if restriction is None:
        restriction = {leaf.name for leaf in tree.get_leaves()}
    result = []
    for node in tree.traverse("preorder"):
        subsplit = []
        for child in node.children:
            clade = set()
            for leaf in child.get_leaves():
                leaf_name = leaf.name  # MK: consider using getattr()
                if leaf_name in restriction:
                    clade.add(leaf.name)
            subsplit.append(tuple(sorted(clade)))
        if len(subsplit) > 1:
            if to_string:
                result.append(subsplit_to_string(tuple(sorted(subsplit))))
            else:
                result.append(tuple(sorted(subsplit)))
    return tuple(result)


def get_subsplits_frozenset(tree):
    result = []
    for node in tree.traverse("preorder"):
        subsplit = []
        for child in node.children:
            clade = set()
            for leaf in child.get_leaves():
                leaf_name = leaf.name  # MK: consider using getattr()
                clade.add(leaf.name)
            subsplit.append(frozenset(clade))
        if len(subsplit) > 1:
            result.append(frozenset(subsplit))
    return frozenset(result)


def get_subsplit_manifest(tree):
    subsplits = get_subsplits_frozenset(tree)
    manifest = {}
    for subsplit in subsplits:
        assert len(subsplit) == 2
        left, right = subsplit
        clade = frozenset(left | right)
        if clade not in manifest:
            manifest[clade] = {frozenset({frozenset(), clade})}
        manifest[clade].add(subsplit)
    return manifest


def get_compatible_subsplits(subsplit1, subsplit2):
    result = []
    if len(subsplit1) < 2 and len(subsplit2) < 2:
        return result
    if len(subsplit1) < 2:
        assert len(subsplit2) == 2
        left2, right2 = subsplit2
        if left2 and right2:
            result.append(subsplit2)
        return result
    if len(subsplit2) < 2:
        assert len(subsplit1) == 2
        left1, right1 = subsplit1
        if left1 and right1:
            result.append(subsplit1)
        return result
    assert len(subsplit1) == 2
    assert len(subsplit2) == 2
    left1, right1 = subsplit1
    left2, right2 = subsplit2
    candidate_cis_left = left1 | left2
    candidate_cis_right = right1 | right2
    if (candidate_cis_left and candidate_cis_right
            and not candidate_cis_left & candidate_cis_right):
        result.append(frozenset({frozenset(candidate_cis_left),
                                 frozenset(candidate_cis_right)}))
    candidate_trans_left = left1 | right2
    candidate_trans_right = right1 | left2
    if (candidate_trans_left and candidate_trans_right
            and not candidate_trans_left & candidate_trans_right):
        result.append(frozenset({frozenset(candidate_trans_left),
                                 frozenset(candidate_trans_right)}))
    return result


def full_cover_restrictions(taxa, ambiguity):
    all_restrictions = combinations(combinations(taxa, len(taxa) - ambiguity), 2)
    for set1, set2 in all_restrictions:
        if set(set1) | set(set2) == set(taxa):
            yield (set1, set2)


def generate_support(tree1, tree2, verbose=False):
    leaf_set1 = get_leaf_set(tree1)
    leaf_set2 = get_leaf_set(tree2)
    full_leaf_set = leaf_set1 | leaf_set2
    manifest1 = get_subsplit_manifest(tree1)
    manifest2 = get_subsplit_manifest(tree2)
    support = []
    clade_stack = [full_leaf_set]
    clades_examined = 0
    while clade_stack:
        clade = clade_stack.pop()
        clades_examined += 1
        restricted_clade1 = frozenset(clade & leaf_set1)
        restricted_clade2 = frozenset(clade & leaf_set2)
        if len(restricted_clade1) == 1:
            leaf = next(iter(restricted_clade1))
            candidate_subsplits1 = [string_to_subsplit(f"{leaf}:")]
        else:
            candidate_subsplits1 = manifest1[restricted_clade1]
        # print(f"1: {candidate_subsplits1}")
        if len(restricted_clade2) == 1:
            leaf = next(iter(restricted_clade2))
            candidate_subsplits2 = [string_to_subsplit(f"{leaf}:")]
        else:
            candidate_subsplits2 = manifest2[restricted_clade2]
        # print(f"2: {candidate_subsplits2}")
        for subsplit_pair in product(candidate_subsplits1, candidate_subsplits2):
            # print(f"pair: {subsplit_pair}")
            compatible_subsplits = get_compatible_subsplits(*subsplit_pair)
            # print(f"compat: {compatible_subsplits}")
            for left, right in compatible_subsplits:
                support.append(frozenset({left, right}))
                if len(left) >= 3:
                    clade_stack.append(left)
                if len(right) >= 3:
                    clade_stack.append(right)
    if verbose:
        print(f"Clades examined: {clades_examined}")
    return support


def support_to_string_set(support):
    string_set = set()
    for subsplit in support:
        string_set.add(subsplit_to_string(subsplit))
    return string_set


def get_manifest_leaf_set(manifest):
    return frozenset().union(*manifest.keys())


def generate_mutual_manifest(manifest1, manifest2, verbose=False):
    leaf_set1 = get_manifest_leaf_set(manifest1)
    leaf_set2 = get_manifest_leaf_set(manifest2)
    full_leaf_set = frozenset(leaf_set1 | leaf_set2)
    mutual_manifest = {}
    clade_stack = [full_leaf_set]
    clades_examined = set()
    while clade_stack:
        clade = clade_stack.pop()
        if clade in clades_examined:
            continue
        clades_examined.add(clade)
        if verbose:
            print(f"clade = {''.join(sorted(clade))}")
        restricted_clade1 = frozenset(clade & leaf_set1)
        restricted_clade2 = frozenset(clade & leaf_set2)
        if verbose:
            print(f"restriction1 = {''.join(sorted(restricted_clade1))}")
            print(f"restriction2 = {''.join(sorted(restricted_clade2))}")
        if len(restricted_clade1) == 0:
            candidate_subsplits1 = [string_to_subsplit(":")]
        elif len(restricted_clade1) == 1:
            leaf = next(iter(restricted_clade1))
            candidate_subsplits1 = [string_to_subsplit(f"{leaf}:")]
        else:
            candidate_subsplits1 = manifest1[restricted_clade1]
        if verbose:
            print(f"[{','.join(subsplit_to_string(subsplit) for subsplit in candidate_subsplits1)}]")
        if len(restricted_clade2) == 0:
            candidate_subsplits2 = [string_to_subsplit(":")]
        elif len(restricted_clade2) == 1:
            leaf = next(iter(restricted_clade2))
            candidate_subsplits2 = [string_to_subsplit(f"{leaf}:")]
        else:
            candidate_subsplits2 = manifest2[restricted_clade2]
        if verbose:
            print(f"[{','.join(subsplit_to_string(subsplit) for subsplit in candidate_subsplits2)}]")
        for subsplit_pair in product(candidate_subsplits1, candidate_subsplits2):
            # print(f"pair: {subsplit_pair}")
            compatible_subsplits = get_compatible_subsplits(*subsplit_pair)
            # print(f"compat: {compatible_subsplits}")
            for left, right in compatible_subsplits:
                if verbose:
                    print(f"{subsplit_to_string(frozenset({left, right}))}")
                if clade not in mutual_manifest:
                    mutual_manifest[clade] = {frozenset({frozenset(), clade})}
                mutual_manifest[clade].add(frozenset({left, right}))
                if len(left) >= 2:
                    clade_stack.append(left)
                if len(right) >= 2:
                    clade_stack.append(right)
    if verbose:
        print(f"Clades examined: {len(clades_examined)}")
    return mutual_manifest


def full_cover_restrictions_n(taxa, ambiguity, n):
    all_restrictions = combinations(combinations(taxa, len(taxa) - ambiguity), n)
    for restriction in all_restrictions:
        if set.union(*(set(clade) for clade in restriction)) == set(taxa):
            yield restriction


def visualize_restricted_rooted_trees_n(taxa, restriction):
    result = {}
    for tree in generate_rooted(taxa):
        projections = tuple(restrict_tree(tree, clade) for clade in restriction)
        signature = tuple(projection.get_topology_id() for projection in projections)
        if signature not in result:
            result[signature] = ProjectionTrees_n(projections, [])
        result[signature].treelist.append(tree)
    return result


def generate_mutual_manifest_n(manifests, verbose=False):
    manifests_iter = iter(manifests)
    mutual_manifest = next(manifests_iter)
    for manifest in manifests_iter:
        mutual_manifest = generate_mutual_manifest(mutual_manifest, manifest, verbose=verbose)
    return mutual_manifest


def merge_manifests(manifests):
    result = {}
    all_keys = set().union(*(set(manifest.keys()) for manifest in manifests))
    for key in all_keys:
        result[key] = frozenset().union(*(manifest.get(key, set()) for manifest in manifests))
    return result


def print_manifest(manifest, show_degenerate=True):
    for key in manifest:
        print(f"{''.join(sorted(key))}: {[subsplit_to_string(subsplit) for subsplit in manifest[key] if show_degenerate or all(subsplit)]}")


def possible_subsplit_sets(manifest, verbose=False):
    leaf_set = get_manifest_leaf_set(manifest)
    tree_list = []
    big_stack = [TreeRecursion(set(), [leaf_set])]
    while big_stack:
        subsplit_set, clade_stack = big_stack.pop()
        if verbose:
            print(f"Current subsplit set: {subsplit_set}")
            print(f"Current clade stack: {clade_stack}")
        possibilities = []
        for clade in clade_stack:
            if verbose:
                print(f"Examining clade: {clade}")
            possibilities.append(subsplit for subsplit in manifest[clade] if all(subsplit))
        for new_subsplits in product(*possibilities):
            if verbose:
                print(f"Examining possibility: {new_subsplits}")
            new_subsplit_set = subsplit_set | set(new_subsplits)
            new_clade_stack = []
            for left, right in new_subsplits:
                if len(left) >= 2:
                    new_clade_stack.append(left)
                    if verbose:
                        print(f"Pushing clade: {left}")
                if len(right) >= 2:
                    new_clade_stack.append(right)
                    if verbose:
                        print(f"Pushing clade: {right}")
            if new_clade_stack:
                big_stack.append(TreeRecursion(new_subsplit_set, new_clade_stack))
                if verbose:
                    print(f"Pushing subsplit set: {new_subsplit_set}")
            else:
                tree_list.append(new_subsplit_set)
                if verbose:
                    print(f"Sending to output: {new_subsplit_set}")
    return tree_list


def possible_single_tree_manifests(manifest, verbose=False):
    leaf_set = get_manifest_leaf_set(manifest)
    tree_list = []
    big_stack = [TreeRecursion({}, [leaf_set])]
    while big_stack:
        current_manifest, clade_stack = big_stack.pop()
        if verbose:
            print(f"Current subsplit set: {current_manifest}")
            print(f"Current clade stack: {clade_stack}")
        possibilities = []
        for clade in clade_stack:
            if verbose:
                print(f"Examining clade: {clade}")
            possibilities.append(subsplit for subsplit in manifest[clade] if all(subsplit))
        for new_subsplits in product(*possibilities):
            if verbose:
                print(f"Examining possibility: {new_subsplits}")
            new_manifest = current_manifest.copy() # | set(new_subsplits)
            new_clade_stack = []
            for subsplit in new_subsplits:
                left, right = subsplit
                clade = frozenset(left | right)
                if clade not in new_manifest:
                    new_manifest[clade] = {frozenset({frozenset(), clade})}
                new_manifest[clade].add(subsplit)
                if len(left) >= 2:
                    new_clade_stack.append(left)
                    if verbose:
                        print(f"Pushing clade: {left}")
                if len(right) >= 2:
                    new_clade_stack.append(right)
                    if verbose:
                        print(f"Pushing clade: {right}")
            if new_clade_stack:
                big_stack.append(TreeRecursion(new_manifest, new_clade_stack))
                if verbose:
                    print(f"Pushing subsplit set: {new_manifest}")
            else:
                tree_list.append(new_manifest)
                if verbose:
                    print(f"Sending to output: {new_manifest}")
    return tree_list


def support_to_manifest(support):
    manifest = {}
    for subsplit in support:
        clade = frozenset().union(*subsplit)
        if clade not in manifest:
            manifest[clade] = {frozenset({frozenset(), clade})}
        manifest[clade].add(subsplit)
    return manifest


def manifest_to_support(manifest):
    support = set()
    for clade in manifest:
        for subsplit in manifest[clade]:
            if all(subsplit):
                support.add(subsplit)
    return support


def manifest_to_tree(manifest):
    leaf_set = get_manifest_leaf_set(manifest)
    tree = Tree()
    clade_stack = [(tree, leaf_set)]
    while clade_stack:
        node, clade = clade_stack.pop()
        possible_subsplits = [subsplit for subsplit in manifest[clade] if all(subsplit)]
        if len(possible_subsplits) > 1:
            print("Multiple trees possible, selecting one randomly")
            subsplit = random.choice(possible_subsplits)
        else:
            subsplit = possible_subsplits[0]
        node.name = subsplit_to_string(subsplit)
        assert len(subsplit) == 2
        left, right = subsplit
        assert len(left) > 0 and len(right) > 0
        if len(left) > 1:
            left_node = node.add_child()
            clade_stack.append((left_node, left))
        else:
            node.add_child(name="".join(left))
        if len(right) > 1:
            right_node = node.add_child()
            clade_stack.append((right_node, right))
        else:
            node.add_child(name="".join(right))
    return tree


def group_trees_by_restriction(trees, taxa, keep_restricted_trees=False):
    result = defaultdict(list)
    if keep_restricted_trees:
        id_to_restricted_tree = {}
    for tree in trees:
        tree_id = tree.get_topology_id()
        restricted_tree = restrict_tree(tree, taxa)
        restricted_id = restricted_tree.get_topology_id()
        result[restricted_id].append(tree_id)
        if keep_restricted_trees:
            id_to_restricted_tree[restricted_id] = restricted_tree
    if keep_restricted_trees:
        return result, id_to_restricted_tree
    else:
        return result




