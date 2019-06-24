import networkx as nx


def edge_in_min_cut(graph, edge):
    edge_weight = graph.get_edge_data(*edge).get('weight', 1)
    min_cut_value, _ = nx.stoer_wagner(graph)

    graph_without_edge = graph.copy()
    graph_without_edge.remove_edge(*edge)
    if nx.is_connected(graph_without_edge):
        min_cut_value_without_edge, _ = nx.stoer_wagner(graph_without_edge)
        return (min_cut_value - min_cut_value_without_edge) >= edge_weight
    else:
        return edge_weight == min_cut_value


def strip_min_cut_edges(graph):
    G = graph.copy()
    for edge in graph.edges():
        if edge_in_min_cut(graph, edge):
            G.remove_edge(*edge)
    return G


