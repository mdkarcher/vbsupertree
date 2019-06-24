from mincutsupertree import *
from small_trees import *
from min_cuts import *


trees = generate_rooted("ABCDE")
tree1 = next(trees)
print(tree1)

print(list(generate_cluster_edges(tree1)))
print(list(generate_proper_cluster_edges(tree1)))

tree2 = next(trees)
print(tree2)

pc_graph = construct_proper_cluster_graph([tree1, tree2])
print(pc_graph.nodes)
print(pc_graph.edges)
for edge in pc_graph.edges:
    print(f"{edge}: {pc_graph.edges[edge]['weight']}")

summ_trees = summarize_trees([tree1, tree2])

contracted_graph = one_pass_contraction(pc_graph, summ_trees)
print(contracted_graph.nodes)
print(contracted_graph.edges)
for edge in contracted_graph.edges:
    print(f"{edge}: {contracted_graph.edges[edge]['weight']}")

disconnected_graph = remove_min_cut_edges_from_proper_cluster_graph(contracted_graph, pc_graph)
print(disconnected_graph.nodes)
print(disconnected_graph.edges)

for comp, subgraph in ((component, disconnected_graph.subgraph(component)) for component in nx.connected_components(disconnected_graph)):
    print(comp)
    print(subgraph.nodes)
    print(subgraph.edges)

foo = mincutsupertree([tree1, tree2])
print(foo)

# Paper example

paper_tree1 = Tree("(((A,B),C),(D,E));")
paper_tree2 = Tree("((A,B),(C,D));")
print(paper_tree1)
print(paper_tree2)
paper_trees = [paper_tree1, paper_tree2]

pc_graph = construct_proper_cluster_graph(paper_trees)
print(pc_graph.nodes)
print(pc_graph.edges)
for edge in pc_graph.edges:
    print(f"{edge}: {pc_graph.edges[edge]['weight']}")

summarized_trees = summarize_trees(paper_trees)
contracted_graph = one_pass_contraction(pc_graph, summarized_trees)
print(contracted_graph.nodes)
print(contracted_graph.edges)
for edge in contracted_graph.edges:
    print(f"{edge}: {contracted_graph.edges[edge]['weight']}")

supertree = mincutsupertree(paper_trees)
print(supertree)

