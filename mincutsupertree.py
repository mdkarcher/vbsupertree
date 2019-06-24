from ete3 import Tree
import networkx as nx
from itertools import combinations
from itertools import chain
from itertools import product
from collections import namedtuple

from min_cuts import *
from small_trees import *


SummarizedTree = namedtuple('SummarizedTree', ['leafset1', 'leafset2', 'weight'])


def generate_cluster_edges(tree):
    leaf_set = {leaf.name for leaf in tree.get_leaves()}
    return combinations(leaf_set, 2)


def generate_proper_cluster_edges(tree):
    if len(tree) < 3:
        return []
    else:
        return chain.from_iterable(generate_cluster_edges(child) for child in tree.get_children())


def construct_proper_cluster_graph(trees, weights=None, leaf_set=None):
    if weights is None:
        weights = [1] * len(trees)
    if leaf_set is None:
        leaf_set = set()
        for tree in trees:
            leaf_set |= {leaf.name for leaf in tree.get_leaves()}
    # TODO: check if tree restriction needs to happen here,
    #  but since it is after the leaf_set union step in the larger
    #  algorithm, I think not
    result = nx.Graph()
    result.add_nodes_from(leaf_set)
    for tree, weight in zip(trees, weights):
        for edge in generate_proper_cluster_edges(tree):
            # print(f"{edge}:")
            if edge not in result.edges:
                # print("Adding edge")
                result.add_edge(*edge, weight=weight)
            else:
                # print("Adding weight")
                result.edges[edge]['weight'] += weight
            # print(f"Current edges: {result.edges}")
    return result


def summarize_tree(tree, weight=1):
    if len(tree) < 2:
        return None
    assert len(tree.children) == 2
    left_child, right_child = tree.children
    left_leafset = {leaf.name for leaf in left_child.get_leaves()}
    right_leafset = {leaf.name for leaf in right_child.get_leaves()}
    return SummarizedTree(left_leafset, right_leafset, weight)


def summarize_trees(trees, weights=None, leaf_restriction=None):
    if weights is None:
        weights = [1] * len(trees)
    result = []
    for tree, weight in zip(trees, weights):
        summary = summarize_tree(tree, weight)
        if summary is not None:
            result.append(summary)
    return result


def get_coincidences(graph, node):
    return {node} | set(graph.nodes[node].get('contraction', set()))


def get_edge_weight(set1, set2, summarized_trees):
    result = 0
    for summarized_tree in summarized_trees:
        if (
                ((set1 & summarized_tree.leafset1) and (set2 & summarized_tree.leafset1))
                or ((set1 & summarized_tree.leafset2) and (set2 & summarized_tree.leafset2))
        ):
            result += summarized_tree.weight
    return result


def one_pass_contraction(graph, summarized_trees):
    max_weight = sum(tree.weight for tree in summarized_trees)
    contracted_graph = graph.copy()
    for edge in graph.edges:
        if (edge in contracted_graph.edges
                and contracted_graph.edges[edge]['weight'] == max_weight):
            contracted_graph = nx.contracted_edge(contracted_graph, edge, self_loops=False)
    for u, v in contracted_graph.edges:
        set1 = get_coincidences(contracted_graph, u)
        set2 = get_coincidences(contracted_graph, v)
        new_weight = get_edge_weight(set1, set2, summarized_trees)
        if new_weight >= max_weight:
            print("Note: Max weight reached during contraction")
        contracted_graph.edges[(u, v)]['weight'] = new_weight
    return contracted_graph


def remove_min_cut_edges_from_proper_cluster_graph(contracted_graph, proper_cluster_graph):
    result = proper_cluster_graph.copy()
    for (u, v) in contracted_graph.edges:
        if edge_in_min_cut(contracted_graph, (u, v)):
            leafset1 = get_coincidences(contracted_graph, u)
            leafset2 = get_coincidences(contracted_graph, v)
            for edge in product(leafset1, leafset2):
                result.remove_edge(*edge)
    return result


def restrict_tree_list(trees, weights, leaf_set):
    restricted_tree_list = []
    restricted_weights = []
    for tree, weight in zip(trees, weights):
        restricted_tree = restrict_tree(tree, leaf_set)
        if restricted_tree is not None:
            restricted_tree_list.append(restricted_tree)
            restricted_weights.append(weight)
    return restricted_tree_list, restricted_weights


def restrict_tree_list_from_component_set(trees, weights, components):
    for leaf_set in components:
        restricted_tree_list, restricted_weights = restrict_tree_list(trees, weights, leaf_set)
        if restricted_tree_list:
            yield restricted_tree_list, restricted_weights
        else:
            print("Note: How did I get here?")


def mincutsupertree(trees, weights=None):
    if weights is None:
        weights = [1] * len(trees)

    leaf_set = set()
    for tree in trees:
        leaf_set |= {leaf.name for leaf in tree.get_leaves()}

    if len(leaf_set) == 1:
        return Tree(f"{''.join(leaf_set)};")
    elif len(leaf_set) == 2:
        return Tree(f"({','.join(leaf_set)});")

    pc_graph = construct_proper_cluster_graph(trees, weights, leaf_set)
    if not nx.is_connected(pc_graph):
        components = nx.connected_components(pc_graph)
    else:
        summarizes_trees = summarize_trees(trees, weights)
        contracted_graph = one_pass_contraction(pc_graph, summarizes_trees)
        disconnected_graph = remove_min_cut_edges_from_proper_cluster_graph(contracted_graph, pc_graph)
        components = nx.connected_components(disconnected_graph)
    subtrees = []
    for restricted_tree_list, restricted_weights in restrict_tree_list_from_component_set(trees, weights, components):
        subtrees.append(mincutsupertree(restricted_tree_list, restricted_weights))
    result = Tree()
    for subtree in subtrees:
        result.add_child(subtree)
    return result





