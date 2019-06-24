from min_cuts import *


# Experiment 0

G = nx.Graph()
G.add_edge('a', 'b', weight=1)
G.add_edge('b', 'c', weight=1)
cut_value, partition = nx.stoer_wagner(G)

# Experiment 1

G = nx.Graph()
G.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])
G.add_edge('a', 'c', weight=2)
edge_in_min_cut(G, ('a', 'b'))
edge_in_min_cut(G, ('b', 'c'))
edge_in_min_cut(G, ('a', 'c'))
edge_in_min_cut(G, ('c', 'd'))

# Experiment 2

G = nx.circular_ladder_graph(5)

# import matplotlib.pyplot as plt
# nx.draw(G)

G.edges()
nx.stoer_wagner(G)
edge_in_min_cut(G, (0, 1))

for edge in G.edges(): print(f"{edge}: {edge_in_min_cut(G, edge)}")

G2 = strip_min_cut_edges(G)

# Experiment 3

G = nx.lollipop_graph(10, 10)
nx.stoer_wagner(G)
edge_in_min_cut(G, (0, 1))
edge_in_min_cut(G, (13, 14))
edge_in_min_cut(G, (18, 19))

for edge in G.edges(): print(f"{edge}: {edge_in_min_cut(G, edge)}")

G2 = strip_min_cut_edges(G)
G2.nodes()
G2.edges()

