from copy import deepcopy
import dendropy as dp
from dendropy.simulate import treesim
from dendropy.datamodel.taxonmodel import Taxon
from dendropy.datamodel.treemodel import Node, Tree
from dendropy.model.discrete import simulate_discrete_char_dataset, Jc69


def taxon_namespace_of_count(taxon_count, taxon_prefix="z"):
    return dp.TaxonNamespace(
        [taxon_prefix + str(idx) for idx in range(taxon_count)]
    )


def kingman(taxon_count, pop_size):
    return treesim.pure_kingman_tree(
        taxon_namespace=taxon_namespace_of_count(taxon_count), pop_size=pop_size
    )


def set_all_branch_lengths_to(tree, length):
    for node in tree:
        node.edge.length = length


def add_outgroup(tree, relative_additional_height):
    desired_height = (
        1 + relative_additional_height
    ) * tree.seed_node.distance_from_tip()

    outgroup = Node(taxon=Taxon("outgroup"), edge_length=desired_height)
    tns = deepcopy(tree.taxon_namespace)
    tns.add_taxon(outgroup.taxon)
    new_root = Node()
    new_root.add_child(outgroup)
    new_root.add_child(tree.seed_node)
    new_tree = Tree(taxon_namespace=tns)
    new_tree.seed_node = new_root
    # Despite my best efforts, I was getting taxon namespace errors. So we round trip
    # from Newick. ¯\_(ツ)_/¯
    new_newick = str(new_tree) + ";"
    return Tree.get(data=new_newick, schema="newick")


def evolve_jc(tree, seq_len):
    return simulate_discrete_char_dataset(seq_len, tree, Jc69()).char_matrices[0]


def alignment_path_of_prefix(prefix):
    return f"{prefix}.fasta"


def simulate(taxon_count, seq_len, prefix, tree_height=None, ingroup_branch_length=None):
    """Simulate a colaescent tree with an outgroup. """
    inner_tree = kingman(taxon_count - 1, pop_size=1)
    if ingroup_branch_length is not None:
        set_all_branch_lengths_to(inner_tree, ingroup_branch_length)
    tree = add_outgroup(inner_tree, 0.0)
    if tree_height is not None:
        tree.scale_edges(tree_height / tree.seed_node.distance_from_tip())
    tree.write(path=f"{prefix}.nwk", schema="newick")
    data = evolve_jc(tree, seq_len)
    data.write(path=alignment_path_of_prefix(prefix), schema="fasta")




