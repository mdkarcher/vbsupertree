import math
import random
import pprint
import collections
from collections import defaultdict
from collections import Counter
from itertools import chain
from itertools import product
from itertools import combinations
from operator import itemgetter
from typing import List, Sequence

import numpy as np
import ete3
import bitarray as ba


class MyTree:
    def __init__(self, tree=None, format=None):
        if tree is None:
            self.tree = ete3.Tree()
        elif isinstance(tree, ete3.Tree):
            self.tree = tree.copy()
        elif isinstance(tree, str):
            if format is None:
                self.tree = ete3.Tree(tree)
            else:
                self.tree = ete3.Tree(tree, format=format)
        elif isinstance(tree, MyTree):
            self.tree = tree.tree.copy()
        else:
            print("Warning: Unknown type for argument 'tree', ignoring")
            self.tree = ete3.Tree()
        self._sort()

    def __eq__(self, other):
        return self.tree.get_topology_id() == other.tree.get_topology_id()

    def __hash__(self):
        return hash(self.tree.get_topology_id())

    def __repr__(self):
        return f"MyTree({repr(self.tree)})"

    def __str__(self):
        return str(self.tree)

    def __copy__(self):
        return MyTree(self.tree)

    def __contains__(self, item):
        if isinstance(item, Clade):
            return self.check_clade(item)
        if isinstance(item, Subsplit):
            return self.check_subsplit(item)
        if isinstance(item, SubsplitClade):
            return self.check_subsplit(item.subsplit)
        if isinstance(item, PCSP):
            return self.check_pcsp(item)
        return item in self.root_clade()

    def copy(self):
        return MyTree(self.tree)

    def refresh_attributes(self):
        for node in self.tree.traverse("postorder"):
            clade = Clade(node.get_leaf_names())
            node.clade = clade
            if node.is_leaf():
                node.name = str(clade)
            else:
                child_clades = [child.clade for child in node.children]
                assert len(child_clades) == 2
                subsplit = Subsplit(*child_clades)
                node.subsplit = subsplit
                node.name = str(subsplit)

    def prune(self, taxon_set):
        taxon_set = Clade(taxon_set)
        new_tree = self.tree.copy()
        new_tree.prune(taxon_set)
        result = MyTree(new_tree)
        result.refresh_attributes()
        return result

    def restrict(self, taxon_set):
        return self.prune(taxon_set)

    def _sort(self):
        tree = self.tree
        for node in tree.traverse("preorder"):
            if node.is_leaf():
                continue
            assert len(node.children) == 2
            left_child, right_child = node.children
            if sorted(right_child.get_leaf_names()) < sorted(left_child.get_leaf_names()):
                node.swap_children()
        return self

    def sort(self):
        result = MyTree(self)
        result._sort()
        return result

    def root_clade(self):
        return Clade(self.tree.get_leaf_names())

    def to_support(self):
        result = SubsplitSupport()
        for node in self.traverse():
            if node.is_leaf():
                continue
            assert len(node.children) == 2
            left_child, right_child = node.children
            left_clade = Clade(left_child.get_leaf_names())
            right_clade = Clade(right_child.get_leaf_names())
            subsplit = Subsplit(left_clade, right_clade)
            result.add(subsplit)
        return result

    def to_pcsp_support(self):
        result = PCSPSupport()
        for node in self.traverse():
            if node.is_leaf():
                continue
            assert len(node.children) == 2
            left_child, right_child = node.children
            left_clade = Clade(left_child.get_leaf_names())
            right_clade = Clade(right_child.get_leaf_names())
            child_subsplit = Subsplit(left_clade, right_clade)
            if node.is_root():
                parent_subsplit = Subsplit(child_subsplit.clade())
            else:
                parent_node = node.up
                left_child, right_child = parent_node.children
                left_clade = Clade(left_child.get_leaf_names())
                right_clade = Clade(right_child.get_leaf_names())
                parent_subsplit = Subsplit(left_clade, right_clade)
            result.add(PCSP(parent_subsplit, child_subsplit))
        return result

    def to_clade_set(self):
        result = set()
        for node in self.traverse():
            if node.is_leaf():
                continue
            clade = Clade(node.get_leaf_names())
            result.add(clade)
        return result

    def iter_clades(self):
        for node in self.traverse():
            if node.is_leaf():
                continue
            yield Clade(node.get_leaf_names())

    def check_clade(self, clade):
        clade = Clade(clade)
        current_clade = self.root_clade()
        current_node = self.tree
        if not current_clade.issuperset(clade):
            return False
        while True:
            if current_clade == clade:
                return True
            suitable_child = False
            for child_node in current_node.children:
                child_clade = Clade(child_node.get_leaf_names())
                if child_clade.issuperset(clade):
                    suitable_child = True
                    current_node = child_node
                    current_clade = child_clade
                    break
            if not suitable_child:
                return False

    def get_clade(self, clade):
        clade = Clade(clade)
        current_clade = self.root_clade()
        current_node = self.tree
        if not current_clade.issuperset(clade):
            return None
        while True:
            if current_clade == clade:
                return MyTree(current_node)
            suitable_child = False
            for child_node in current_node.children:
                child_clade = Clade(child_node.get_leaf_names())
                if child_clade.issuperset(clade):
                    suitable_child = True
                    current_node = child_node
                    current_clade = child_clade
                    break
            if not suitable_child:
                return None

    def check_subsplit(self, subsplit):
        clade = subsplit.clade()
        if subsplit.is_trivial():
            return self.check_clade(clade)
        subtree = self.get_clade(clade)
        if subtree is None:
            return False
        child_clades = [Clade(child.get_leaf_names()) for child in subtree.tree.children]
        if len(child_clades) != 2:
            return False
        real_subsplit = Subsplit(*child_clades)
        return subsplit == real_subsplit

    def check_pcsp(self, pcsp):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not of type PCSP.")
        return self.check_subsplit(pcsp.parent) and self.check_subsplit(pcsp.child)

    def traverse(self, strategy='levelorder'):
        return self.tree.traverse(strategy=strategy)

    @staticmethod
    def _node_to_subsplit(node: ete3.Tree):
        assert len(node.children) == 2
        left_child, right_child = node.children
        return Subsplit(left_child.get_leaf_names(), right_child.get_leaf_names())

    @staticmethod
    def _node_to_pcsp(node: ete3.Tree):
        assert len(node.children) == 2
        child = MyTree._node_to_subsplit(node)
        if node.is_root():
            parent = Subsplit(node.get_leaf_names())
        else:
            parent = MyTree._node_to_subsplit(node.up)
        return PCSP(parent, child)

    def traverse_subsplits(self, strategy='levelorder'):
        for node in self.tree.traverse(strategy):
            if not node.is_leaf():
                yield self._node_to_subsplit(node)

    def traverse_pcsps(self, strategy='levelorder'):
        for node in self.tree.traverse(strategy):
            if not node.is_leaf():
                yield self._node_to_pcsp(node)

    @classmethod
    def _generate_unrooted(cls, taxa):
        if len(taxa) <= 3:
            # return [Tree('(' + ','.join(taxa) + ');')]
            yield ete3.Tree('(' + ','.join(taxa) + ');')
        else:
            # res = []
            sister = ete3.Tree('(' + taxa[-1] + ');')
            for tree in MyTree._generate_unrooted(taxa[:-1]):
                for node in tree.traverse('preorder'):
                    if not node.is_root():
                        node.up.add_child(sister)
                        node.detach()
                        sister.add_child(node)
                        yield tree.copy()
                        node.detach()
                        sister.up.add_child(node)
                        sister.detach()

    @classmethod
    def generate_unrooted(cls, taxa):
        for tree in cls._generate_unrooted(taxa):
            yield MyTree(tree)

    @classmethod
    def _generate_rooted(cls, taxa):
        if len(taxa) == 1:
            yield ete3.Tree(f"{taxa[0]};")
        elif len(taxa) == 2:
            yield ete3.Tree(f"({','.join(taxa)});")
        else:
            for tree in cls._generate_unrooted(taxa):
                for node in tree.traverse('preorder'):
                    if not node.is_root():
                        tree.set_outgroup(node)
                        yield tree.copy()

    @classmethod
    def generate_rooted(cls, taxa):
        for tree in cls._generate_rooted(taxa):
            yield MyTree(tree)

    @staticmethod
    def random(taxa, rooted=True):
        tree = ete3.Tree()
        tree.populate(len(taxa), taxa)
        if not rooted:
            tree.unroot()
        return MyTree(tree)

    @staticmethod
    def _node_clade(node):
        return Clade(node.get_leaf_names())

    def clade_score(self, clade_probs):
        result = 1.0
        for clade in self.to_clade_set():
            result *= clade_probs[clade]
        return result

    def rename_tips(self, map_dict, refresh_attributes=True):
        for tip in self.tree.iter_leaves():
            if tip.name in map_dict:
                tip.name = map_dict[tip.name]
        if refresh_attributes:
            self.refresh_attributes()


class Clade(frozenset):
    def __new__(cls, arg=None):
        if arg is None:
            arg = frozenset()
        return super(Clade, cls).__new__(cls, arg)

    def __lt__(self, other):
        return sorted(self) < sorted(other)

    def __gt__(self, other):
        return sorted(self) > sorted(other)

    def __le__(self, other):
        return sorted(self) <= sorted(other)

    def __ge__(self, other):
        return sorted(self) >= sorted(other)

    def __repr__(self):
        result = "Clade()"
        if self:
            contents = ", ".join(map(repr, sorted(self)))
            result = f"Clade({{{contents}}})"
        return result

    def __str__(self):
        return "".join(map(str, sorted(self)))

    def __and__(self, other):
        return Clade(super().__and__(other))

    def __or__(self, other):
        return Clade(super().__or__(other))

    def __xor__(self, other):
        return Clade(super().__xor__(other))

    def __sub__(self, other):
        return Clade(super().__sub__(other))

    def to_bitarray(self, bit_map):
        return ba.bitarray(item in self for item in bit_map)


class Subsplit:
    def __init__(self, clade1=None, clade2=None):
        clade1 = Clade(clade1)
        clade2 = Clade(clade2)
        self.clade1 = min(clade1, clade2)
        self.clade2 = max(clade1, clade2)
        self.data = frozenset({self.clade1, self.clade2})

    def __repr__(self):
        return f"Subsplit({repr(self.clade1)}, {repr(self.clade2)})"

    def __str__(self):
        return f"{str(self.clade1)}:{str(self.clade2)}"

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __contains__(self, item):
        if not isinstance(item, Clade):
            raise TypeError("Argument item not of type Clade.")
        return item == self.clade1 or item == self.clade2

    def __len__(self):
        return len(self.clade())

    def children(self):
        yield self.clade1
        yield self.clade2

    def clade(self):
        return Clade(self.clade1 | self.clade2)

    def compatible_child(self, item):
        if not isinstance(item, Clade) and not isinstance(item, Subsplit):
            raise TypeError("Argument 'item' not of class Clade or Subsplit.")
        if isinstance(item, Subsplit):
            clade = item.clade()
        else:
            clade = item
        if clade.issubset(self.clade1):
            return self.clade1
        elif clade.issubset(self.clade2):
            return self.clade2
        else:
            return None

    def cross(self, other):
        return self.union_cis(other), self.union_trans(other)

    def is_strictly_valid(self):
        return self.is_valid() and not self.is_trivial()

    def is_trivial(self):
        return (not self.clade1) or (not self.clade2)

    def is_valid(self):
        return not (self.clade1 & self.clade2)

    def nontrivial_children(self):
        if len(self.clade1) > 1:
            yield self.clade1
        if len(self.clade2) > 1:
            yield self.clade2

    def restrict(self, taxon_set):
        taxon_set = set(taxon_set)
        return Subsplit(self.clade1 & taxon_set, self.clade2 & taxon_set)

    def to_bitarray(self, bit_map, swap=False):
        if swap:
            result = self.clade2.to_bitarray(bit_map) + self.clade1.to_bitarray(bit_map)
        else:
            result = self.clade1.to_bitarray(bit_map) + self.clade2.to_bitarray(bit_map)
        return result

    def union_cis(self, other):
        return Subsplit(self.clade1 | other.clade1, self.clade2 | other.clade2)

    def union_trans(self, other):
        return Subsplit(self.clade1 | other.clade2, self.clade2 | other.clade1)

    def valid_ancestor(self, other):
        if not isinstance(other, Subsplit):
            raise TypeError("Argument 'other' is not of type Subsplit.")
        clade = self.clade()
        return clade.issubset(other.clade1) or clade.issubset(other.clade2)

    def valid_child(self, other):
        if not isinstance(other, Subsplit):
            raise TypeError("Argument 'other' is not of type Subsplit.")
        return other.clade() in set(self.children())

    def valid_descendant(self, other):
        if not isinstance(other, Subsplit):
            raise TypeError("Argument 'other' is not of type Subsplit.")
        clade = other.clade()
        return clade.issubset(self.clade1) or clade.issubset(self.clade2)

    def valid_parent(self, other):
        if not isinstance(other, Subsplit):
            raise TypeError("Argument 'other' is not of type Subsplit.")
        return self.clade() in set(other.children())

    @staticmethod
    def from_node(node):
        assert isinstance(node, ete3.TreeNode)
        if node.is_leaf() or len(node.children) < 2:
            return Subsplit(Clade(node.get_leaf_names()))
        assert len(node.children) == 2
        left_child, right_child = node.children
        return Subsplit(Clade(left_child.get_leaf_names()), Clade(right_child.get_leaf_names()))

    @staticmethod
    def compatible_subsplits(clade):
        clade = Clade(clade)
        size = len(clade)
        if size < 2:
            raise StopIteration
        for k in range(1, size//2 + 1):
            for left in combinations(clade, k):
                left_clade = Clade(left)
                right_clade = clade-left_clade
                if k == size / 2 and right_clade > left_clade:
                    continue
                yield Subsplit(left_clade, right_clade)

    @staticmethod
    def cross_multiple(subsplits):
        subsplits = list(subsplits)
        n = len(subsplits)
        sigs = product(*[[False, True]] * (n-1))
        result = []
        for sig in sigs:
            ss = subsplits[0]
            for trans, subsplit in zip(sig, subsplits[1:]):
                if trans:
                    ss = ss.union_trans(subsplit)
                else:
                    ss = ss.union_cis(subsplit)
            if ss.is_strictly_valid():
                result.append(ss)
        return result

    @staticmethod
    def cross_multiple_naive(subsplits):
        n = len(subsplits)
        sigs = product(*[[False, True]] * n)
        result = []
        for sig in sigs:
            clade1 = Clade()
            clade2 = Clade()
            for trans, subsplit in zip(sig, subsplits):
                if trans:
                    clade1 |= subsplit.clade2
                    clade2 |= subsplit.clade1
                else:
                    clade1 |= subsplit.clade1
                    clade2 |= subsplit.clade2
            ss = Subsplit(clade1, clade2)
            if ss.is_strictly_valid():
                result.append(ss)
        return result

    @staticmethod
    def parent_from_node(node):
        assert isinstance(node, ete3.TreeNode)
        if node.is_root():
            return Subsplit(Clade(node.get_leaf_names()))
        return Subsplit.from_node(node.up)


class SubsplitClade:
    def __init__(self, subsplit: Subsplit, clade: Clade):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not of class Subsplit.")
        if not isinstance(clade, Clade):
            raise TypeError("Argument 'clade' not of class Clade.")
        if clade not in subsplit:
            raise ValueError("Argument 'clade' not a valid child of argument 'subsplit'.")
        self.subsplit = subsplit
        self.clade = clade

    def __eq__(self, other):
        if not isinstance(other, SubsplitClade):
            return False
        return self.subsplit == other.subsplit and self.clade == other.clade

    def __hash__(self):
        return hash((self.subsplit, self.clade))

    def __repr__(self):
        return f"SubsplitClade({repr(self.subsplit)}, {repr(self.clade)})"

    def __str__(self):
        sister = self.subsplit.clade1 if self.subsplit.clade2 == self.clade else self.subsplit.clade2
        return f"{str(sister)}/{str(self.clade)}"

    @property
    def other_clade(self):
        other = self.subsplit.clade1
        if other == self.clade:
            other = self.subsplit.clade2
        return other

    def restrict(self, taxon_set):
        taxon_set = Clade(taxon_set)
        restricted_subsplit = self.subsplit.restrict(taxon_set)
        restricted_clade = Clade(self.clade & taxon_set)
        return SubsplitClade(subsplit=restricted_subsplit, clade=restricted_clade)

    def to_bitarray(self, bit_map):
        return self.subsplit.to_bitarray(bit_map=bit_map, swap=(self.clade == self.subsplit.clade1))


class PCSP:
    def __init__(self, parent, child):
        if not (isinstance(parent, Subsplit) or isinstance(parent, Clade) or isinstance(parent, SubsplitClade)):
            raise TypeError("Argument 'parent' not a Subsplit, Clade, or SubsplitClade.")
        if not isinstance(child, Subsplit):
            raise TypeError("Argument 'child' not a Subsplit.")
        if isinstance(parent, Clade):
            parent = Subsplit(parent)
        if isinstance(parent, SubsplitClade):
            if parent.clade != child.clade():
                raise ValueError("Child clade of argument 'parent' not equal to parent clade of argument 'child'.")
            parent = parent.subsplit
        self.parent = parent
        self.child = child

    def __repr__(self):
        return f"PCSP({repr(self.parent)}, {repr(self.child)})"

    def __str__(self):
        return f"{str(self.parent)}, {str(self.child)}"

    def __eq__(self, other):
        if not isinstance(other, PCSP):
            return False
        return self.parent == other.parent and self.child == other.child

    def __hash__(self):
        return hash((self.parent, self.child))

    def check(self):
        return self.parent.is_valid() and self.child.is_valid() and self.parent.valid_child(self.child)

    def is_valid(self):
        return self.check()

    def parent_clade(self):
        return SubsplitClade(self.parent, self.child.clade())

    def restrict(self, taxon_set):
        taxon_set = set(taxon_set)
        restricted_parent = self.parent.restrict(taxon_set)
        restricted_child = self.child.restrict(taxon_set)
        return PCSP(restricted_parent, restricted_child)

    def to_bitarray(self, bit_map):
        child_ba = min(self.child.clade1.to_bitarray(bit_map),
                       self.child.clade2.to_bitarray(bit_map))
        return self.parent_clade().to_bitarray(bit_map) + child_ba

    @staticmethod
    def from_node(node):
        assert isinstance(node, ete3.TreeNode)
        return PCSP(Subsplit.parent_from_node(node), Subsplit.from_node(node))


# TODO: eliminate include_trivial option, make code do the correct thing
#  in the appropriate contexts (including trivial subsplits while mutualizing,
#  ignoring them most of the rest of the time.
class SubsplitSupport:
    def __init__(self, arg=None, include_trivial=False):
        self.data = defaultdict(set)
        self.include_trivial = include_trivial
        self.update(arg)

    def __repr__(self):
        pretty_repr = pprint.pformat(self.to_set())
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join(("SubsplitSupport(", pretty_repr, ")"))

    def __str__(self):
        return str(self.to_string_dict())
        # return pprint.pformat(set(map(str, self.to_set())))

    def __iter__(self):
        yield from iter(self.to_set())

    def __getitem__(self, item):
        return self.data.get(item, set())

    def __eq__(self, other):
        if not isinstance(other, SubsplitSupport):
            return False
        return self.to_set() == other.to_set()

    def __len__(self):
        result = 0
        for clade in self.data:
            result += len(self.data[clade])
        return result

    def add(self, subsplit):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Error: Argument subsplit not class Subsplit")
        if not subsplit.is_strictly_valid():
            # print(f"Warning: subsplit ({str(subsplit)}) is not strictly valid, skipping")
            return
        self.data[subsplit.clade()].add(subsplit)
        # if self.include_trivial or not subsplit.is_trivial():
        #     self.data[subsplit.clade()].add(subsplit)
        #     if self.include_trivial:
        #         self.data[subsplit.clade()].add(Subsplit(subsplit.clade()))

    def update(self, arg):
        if arg is None:
            return
        if isinstance(arg, Subsplit):
            arg = [arg]
        if isinstance(arg, SubsplitSupport):
            arg = arg.to_set()
        try:
            for subsplit in arg:
                if not isinstance(subsplit, Subsplit):
                    print(f"{subsplit} is not a Subsplit, skipping")
                    continue
                if not subsplit.is_strictly_valid():
                    # print(f"{subsplit} is not strictly valid, skipping")
                    continue
                self.data[subsplit.clade()].add(subsplit)
                # if self.include_trivial or not subsplit.is_trivial():
                #     self.data[subsplit.clade()].add(subsplit)
                #     if self.include_trivial:
                #         self.data[subsplit.clade()].add(Subsplit(subsplit.clade()))
        except TypeError:
            print(f"Error: argument 'arg' is not Subsplit or Iterable.")

    def clades(self):
        return self.data.keys()

    def get_taxon_set(self):
        return set().union(*self.data)

    def root_clade(self):
        return Clade(self.get_taxon_set())

    def is_complete(self, verbose=False):
        root_clade = self.root_clade()
        if root_clade not in self.data:
            if verbose:
                print(root_clade)
            return False
        clade_stack = [root_clade]
        visited_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in visited_clades:
                continue
            visited_clades.add(clade)
            subsplits = self.data[clade]
            if not subsplits:
                if verbose:
                    print(clade)
                return False
            for subsplit in subsplits:
                for child in subsplit.nontrivial_children():
                    clade_stack.append(child)
        return True

    # def complete_subset(self):
    #     root_clade = Clade(self.get_taxon_set())
    #     if root_clade not in self.data:
    #         print(f"Error: No possible root splits.")
    #         return None
    #     clade_stack = [root_clade]
    #     result = SubsplitSupport(include_trivial=self.include_trivial)
    #     while clade_stack:
    #         clade = clade_stack.pop()
    #         subsplits = self.data[clade]
    #         for subsplit in subsplits:
    #             result.add(subsplit)
    #             for child in subsplit.nontrivial_children():
    #                 clade_stack.append(child)
    #     return result

    def restrict(self, taxon_set):
        result = SubsplitSupport()
        for clade in self.data:
            result.update(subsplit.restrict(taxon_set) for subsplit in self.data[clade])
        return result

    def to_set(self):
        result = set()
        for clade in self.data:
            result.update(self.data[clade])
        return result

    def to_string_dict(self):
        return {str(clade): set(map(str, self.data[clade])) for clade in self.data}

    def to_string_set(self):
        return set(chain.from_iterable(map(str, self.data[clade]) for clade in self.data))

    def random_tree(self):
        # if not self.is_complete():
        #     print("Error: SubsplitSupport not complete, no tree possible.")
        #     return
        root_clade = self.root_clade()
        tree = MyTree()
        clade_stack = [(tree.tree, root_clade)]  # MyTree alteration
        while clade_stack:
            node, clade = clade_stack.pop()
            possible_subsplits = [subsplit for subsplit in self.data[clade] if not subsplit.is_trivial()]
            # assert len(possible_subsplits) > 0
            if len(possible_subsplits) < 1:
                raise ValueError(f"Error: support not complete, no children for clade {str(clade)}")
            subsplit = random.choice(possible_subsplits)
            node.name = str(subsplit)
            # assert len(subsplit) == 2
            left = subsplit.clade1
            right = subsplit.clade2
            assert len(left) > 0 and len(right) > 0
            if len(left) > 1:
                left_node = node.add_child()
                clade_stack.append((left_node, left))
            else:
                node.add_child(name=str(left))
            if len(right) > 1:
                right_node = node.add_child()
                clade_stack.append((right_node, right))
            else:
                node.add_child(name=str(right))
        return tree

    def all_strictly_complete_supports(self, verbose=False):
        # if not self.is_complete():
        #     print("Error: SubsplitSupport not complete, no tree possible.")
        #     return
        root_clade = self.root_clade()
        support_list = []
        big_stack = [(SubsplitSupport(), [root_clade])]
        while big_stack:
            current_support, clade_stack = big_stack.pop()
            if verbose:
                print(f"Current subsplit set: {current_support}")
                print(f"Current clade stack: {clade_stack}")
            possibilities = []
            for clade in clade_stack:
                if verbose:
                    print(f"Examining clade: {clade}")
                possibilities.append(subsplit for subsplit in self.data[clade] if not subsplit.is_trivial())
            for new_subsplits in product(*possibilities):
                if verbose:
                    print(f"Examining possibility: {new_subsplits}")
                new_support = SubsplitSupport(current_support)
                new_clade_stack = []
                for subsplit in new_subsplits:
                    # left, right = subsplit
                    # clade = subsplit.clade()
                    # if clade not in new_support:
                    #     new_support[clade] = {frozenset({frozenset(), clade})}
                    # new_support[clade].add(subsplit)
                    new_support.add(subsplit)
                    for child_clade in subsplit.nontrivial_children():
                        new_clade_stack.append(child_clade)
                        if verbose:
                            print(f"Pushing clade: {child_clade}")
                if new_clade_stack:
                    big_stack.append((new_support, new_clade_stack))
                    if verbose:
                        print(f"Pushing subsplit set: {new_support}")
                else:
                    support_list.append(new_support)
                    if verbose:
                        print(f"Sending to output: {new_support}")
        return support_list

    def all_trees(self):
        strict_supports = self.all_strictly_complete_supports()
        return [support.random_tree() for support in strict_supports]

    def mutualize(self, other):
        if not isinstance(other, SubsplitSupport):
            raise TypeError("Argument 'other' not instance of SubsplitSupport.")
        assert isinstance(other, SubsplitSupport)
        result = SubsplitSupport()
        root_clade1 = self.root_clade()
        root_clade2 = other.root_clade()
        big_clade = root_clade1 | root_clade2
        clade_stack = [big_clade]
        examined_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in examined_clades:
                continue
            examined_clades.add(clade)
            restricted_clade1 = clade & root_clade1
            restricted_clade2 = clade & root_clade2
            candidate_subsplits1 = self[restricted_clade1] | {Subsplit(restricted_clade1)}
            candidate_subsplits2 = other[restricted_clade2] | {Subsplit(restricted_clade2)}
            candidate_big_subsplits = chain.from_iterable(
                (big_subsplit for big_subsplit in subsplit1.cross(subsplit2) if big_subsplit.is_strictly_valid())
                for subsplit1, subsplit2 in product(candidate_subsplits1, candidate_subsplits2)
            )
            for big_subsplit in candidate_big_subsplits:
                result.add(big_subsplit)
                for child_clade in big_subsplit.nontrivial_children():
                    clade_stack.append(child_clade)
        return result

    def add_tree(self, tree):
        for node in tree.traverse():
            if node.is_leaf():
                continue
            assert len(node.children) == 2
            left_child, right_child = node.children
            left_clade = Clade(left_child.get_leaf_names())
            right_clade = Clade(right_child.get_leaf_names())
            subsplit = Subsplit(left_clade, right_clade)
            self.add(subsplit)

    @staticmethod
    def from_tree(tree):
        # if isinstance(tree, MyTree):
        #     tree = tree.tree
        result = SubsplitSupport()
        result.add_tree(tree)
        return result

    @staticmethod
    def from_trees(trees):
        # if isinstance(tree, MyTree):
        #     tree = tree.tree
        result = SubsplitSupport()
        for tree in trees:
            result.add_tree(tree)
        return result

    @staticmethod
    def common_support(supports):
        result = SubsplitSupport()
        root_clades = [set(support.root_clade()) for support in supports]
        big_clade = Clade(set.union(*root_clades))
        clade_stack = [big_clade]
        examined_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in examined_clades:
                continue
            examined_clades.add(clade)
            restricted_clades = [clade & root_clade for root_clade in root_clades]
            candidate_subsplits_list = [support[restricted_clade] | {Subsplit(restricted_clade)} for
                                        support, restricted_clade in zip(supports, restricted_clades)]
            candidate_big_subsplits = chain.from_iterable(
                Subsplit.cross_multiple(candidate_subsplits) for candidate_subsplits in
                product(*candidate_subsplits_list)
            )
            for big_subsplit in candidate_big_subsplits:
                result.add(big_subsplit)
                for child_clade in big_subsplit.nontrivial_children():
                    clade_stack.append(child_clade)
        return result


class PCSPSupport:
    def __init__(self, arg=None):
        self.data = dict()
        self.update(arg)

    def __copy__(self):
        return PCSPSupport(self.to_set())

    def __repr__(self):
        pretty_repr = pprint.pformat(self.to_set())
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join(("PCSPSupport(", pretty_repr, ")"))

    def __str__(self):
        return str(self.to_string_dict())
        # return pprint.pformat(set(map(str, self.to_set())))

    def __iter__(self):
        yield from iter(self.to_set())

    def __getitem__(self, parent):
        if isinstance(parent, SubsplitClade):
            return self.data[parent]
        if isinstance(parent, Subsplit):
            return tuple(self.data[SubsplitClade(parent, child)] for child in parent.nontrivial_children())
        raise TypeError("Argument 'parent' not SubsplitClade or Subsplit.")

    def __eq__(self, other):
        if not isinstance(other, PCSPSupport):
            return False
        return self.to_set() == other.to_set()

    def __len__(self):
        result = 0
        for parent_clade, children in self.data.items():
            result += len(children)
        return result

    def add(self, pcsp):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        if not pcsp.is_valid():
            print(f"Warning: argument pcsp ({str(pcsp)}) is not valid, skipping")
            return
        parent_clade = pcsp.parent_clade()
        if parent_clade not in self.data:
            self.data[parent_clade] = set()
        self.data[parent_clade].add(pcsp.child)

    def copy(self):
        return self.__copy__()

    def get(self, parent, default=None):
        if isinstance(parent, SubsplitClade):
            return self.data.get(parent, default)
        if isinstance(parent, Subsplit):
            return tuple(self.data.get(SubsplitClade(parent, child), default) for child in parent.nontrivial_children())
        raise TypeError("Argument 'parent' not SubsplitClade or Subsplit.")

    def update(self, arg):
        if arg is None:
            return
        if isinstance(arg, PCSP):
            arg = [arg]
        if isinstance(arg, PCSPSupport):
            arg = arg.to_set()
        try:
            for pcsp in arg:
                if not isinstance(pcsp, PCSP):
                    print(f"{pcsp} is not a PCSP, skipping")
                    continue
                self.add(pcsp)
        except TypeError:
            print(f"Error: {arg} is not PCSP or Iterable.")

    def get_taxon_set(self):
        result = set().union(*(parent_clade.subsplit.clade() for parent_clade in self.data))
        # for parent_clade in self.data:
        #     result |= parent_clade.subsplit.clade()
        return result

    def root_clade(self):
        return Clade(self.get_taxon_set())

    def root_subsplit(self):
        return Subsplit(self.root_clade())

    def root_subsplit_clade(self):
        root_clade = self.root_clade()
        return SubsplitClade(Subsplit(root_clade), Clade(root_clade))

    def is_complete(self, verbose=False):
        result = True
        root_subsplit_clade = self.root_subsplit_clade()
        if root_subsplit_clade not in self.data:
            if verbose:
                print("No root subsplits found.")
                result = False
            else:
                return False
        parent_clade_stack = [root_subsplit_clade]
        visited_parent_clades = set()
        while parent_clade_stack:
            parent_clade = parent_clade_stack.pop()
            if len(parent_clade.clade) < 2:
                continue
            if parent_clade in visited_parent_clades:
                continue
            visited_parent_clades.add(parent_clade)
            mini_support = self.data.get(parent_clade, set())
            if not mini_support:
                if verbose:
                    print(f"Not locally complete: {mini_support}")
                    result = False
                else:
                    return False
            for subsplit in mini_support:
                for child in subsplit.nontrivial_children():
                    parent_clade_stack.append(SubsplitClade(subsplit, child))
        return result

    # Keep an eye on if this does what other functions need.
    def valid_parent(self, parent, subsplit_set=None):
        if subsplit_set is None:
            subsplit_set = self.to_subsplit_set()
        if not isinstance(parent, SubsplitClade):
            raise TypeError("Argument 'parent' is not of type SubsplitClade.")

        result = False
        if (len(parent.clade) == 0
                or (len(parent.clade) == 1 and parent.clade.issubset(self.root_clade()))
                or parent.subsplit in subsplit_set):
            result = True
        return result

    def restrict(self, taxon_set):
        taxon_set = Clade(taxon_set)
        root_subsplit_clade = self.root_subsplit_clade()
        result = PCSPSupport()
        parent_clade_stack = [(root_subsplit_clade, root_subsplit_clade.restrict(taxon_set))]
        visited = set()
        while parent_clade_stack:
            parent_clade, last_valid_parent_clade = parent_clade_stack.pop()
            if (parent_clade, last_valid_parent_clade) in visited:
                continue
            visited.add((parent_clade, last_valid_parent_clade))
            for child in self.data[parent_clade]:
                restricted_child = child.restrict(taxon_set)
                if not restricted_child.is_trivial():
                    result.add(PCSP(last_valid_parent_clade, restricted_child))
                    if len(child.clade1 & taxon_set) > 1:
                        parent_clade_stack.append(
                            (SubsplitClade(child, child.clade1),
                             SubsplitClade(restricted_child, child.clade1 & taxon_set))
                        )
                    if len(child.clade2 & taxon_set) > 1:
                        parent_clade_stack.append(
                            (SubsplitClade(child, child.clade2),
                             SubsplitClade(restricted_child, child.clade2 & taxon_set))
                        )
                else:
                    if len(child.clade1 & taxon_set) > 1:
                        parent_clade_stack.append(
                            (SubsplitClade(child, child.clade1),
                             last_valid_parent_clade)
                        )
                    if len(child.clade2 & taxon_set) > 1:
                        parent_clade_stack.append(
                            (SubsplitClade(child, child.clade2),
                             last_valid_parent_clade)
                        )
        # for parent in self.data:
        #     result.update(pcsp.restrict(taxon_set) for pcsp in self[parent].to_pcsp_set())
        return result

    def to_set(self):
        result = set()
        for parent_clade in self.data:
            result.update(PCSP(parent_clade, child) for child in self.data[parent_clade])
        return result

    def to_string_dict(self):
        return {str(parent_clade): set(map(str, self.data[parent_clade])) for parent_clade in self.data}

    def to_string_set(self):
        return set(map(str, self.to_set()))

    def to_subsplit_set(self):
        result = set()
        for parent_clade, children in self.data.items():
            result.add(parent_clade.subsplit)
            result.update(children)
        return result

    def random_tree(self):
        if not self.is_complete():
            raise ValueError("PCSPSupport not complete, no tree possible.")
        root_subsplit_clade = self.root_subsplit_clade()
        tree = MyTree()
        parent_clade_stack = [(tree.tree, root_subsplit_clade)]
        while parent_clade_stack:
            node, parent_clade = parent_clade_stack.pop()
            possible_subsplits = [subsplit for subsplit in self.data[parent_clade]
                                  if not subsplit.is_trivial()]
            assert possible_subsplits
            subsplit = random.choice(possible_subsplits)
            node.name = str(subsplit)
            left = subsplit.clade1
            right = subsplit.clade2
            assert len(left) > 0 and len(right) > 0
            if len(left) > 1:
                left_node = node.add_child()
                parent_clade_stack.append((left_node, SubsplitClade(subsplit, left)))
            else:
                node.add_child(name=str(left))
            if len(right) > 1:
                right_node = node.add_child()
                parent_clade_stack.append((right_node, SubsplitClade(subsplit, right)))
            else:
                node.add_child(name=str(right))
        return tree

    def all_strictly_complete_supports(self, verbose=False):
        # if not self.is_complete():
        #     print("Error: SubsplitSupport not complete, no tree possible.")
        #     return
        root_subsplit_clade = self.root_subsplit_clade()
        support_list = []
        big_stack = [(PCSPSupport(), [root_subsplit_clade])]
        while big_stack:
            current_support, parent_clade_stack = big_stack.pop()
            if verbose:
                print(f"Current PCSP set: {current_support}")
                print(f"Current parent-clade stack: {parent_clade_stack}")
            possibilities = []
            for parent_clade in parent_clade_stack:
                if verbose:
                    print(f"Examining clade: {parent_clade}")
                subpossibilities = []
                for subsplit in self.data[parent_clade]:
                    if subsplit.is_trivial():
                        continue
                    pcsp = PCSP(parent_clade, subsplit)
                    if verbose:
                        print(f" Appending PCSP: {pcsp}")
                    subpossibilities.append(pcsp)
                possibilities.append(subpossibilities)
                # possibilities.append(PCSP(parent_clade, subsplit) for subsplit in self.data[parent_clade] if not subsplit.is_trivial())
            for new_pcsps in product(*possibilities):
                if verbose:
                    print(f"Examining possibility: {new_pcsps}")
                new_support = PCSPSupport(current_support)
                new_parent_clade_stack = []
                for pcsp in new_pcsps:
                    # left, right = subsplit
                    # clade = subsplit.clade()
                    # if clade not in new_support:
                    #     new_support[clade] = {frozenset({frozenset(), clade})}
                    # new_support[clade].add(subsplit)
                    new_support.add(pcsp)
                    child = pcsp.child
                    for child_clade in child.nontrivial_children():
                        child_ss_clade = SubsplitClade(child, child_clade)
                        new_parent_clade_stack.append(child_ss_clade)
                        if verbose:
                            print(f"Pushing child-clade: {child_ss_clade}")
                if new_parent_clade_stack:
                    big_stack.append((new_support, new_parent_clade_stack))
                    if verbose:
                        print(f"Pushing subsplit set: {new_support}")
                else:
                    support_list.append(new_support)
                    if verbose:
                        print(f"Sending to output: {new_support}")
        return support_list

    def all_trees(self, verbose=False):
        strict_supports = self.all_strictly_complete_supports(verbose=verbose)
        return (support.random_tree() for support in strict_supports)

    def mutualize(self, other, verbose=False, visited=None, stats=None):
        if not isinstance(other, PCSPSupport):
            raise TypeError("Argument 'other' not PCSPSupport.")
        if isinstance(visited, set):
            visited.clear()
        else:
            visited = set()
        result = PCSPSupport()
        self_subsplit_set = self.to_subsplit_set()
        other_subsplit_set = other.to_subsplit_set()
        root_subsplit_clade1 = self.root_subsplit_clade()
        root_subsplit_clade2 = other.root_subsplit_clade()
        root_clade1 = root_subsplit_clade1.clade
        root_clade2 = root_subsplit_clade2.clade
        big_clade = root_clade1 | root_clade2
        big_root_parent_clade = SubsplitClade(Subsplit(big_clade), Clade(big_clade))
        parent_clade_stack = [(big_root_parent_clade, root_subsplit_clade1, root_subsplit_clade2)]
        while parent_clade_stack:
            triplet = parent_clade_stack.pop()
            if triplet in visited:
                if stats is not None:
                    stats["skipped_due_to_visited"] = stats.get("skipped_due_to_visited", 0) + 1
                continue
            if stats is not None:
                stats["visited_triplets"] = stats.get("visited_triplets", 0) + 1
            visited.add(triplet)
            parent_clade, restricted_parent_subsplit_clade1, restricted_parent_subsplit_clade2 = triplet
            if verbose:
                print(f"Examining ({parent_clade}, {restricted_parent_subsplit_clade1}, {restricted_parent_subsplit_clade2})")
            candidate_subsplits1 = self.get(restricted_parent_subsplit_clade1, set()) | {Subsplit(restricted_parent_subsplit_clade1.clade)}
            candidate_subsplits2 = other.get(restricted_parent_subsplit_clade2, set()) | {Subsplit(restricted_parent_subsplit_clade2.clade)}
            if verbose:
                print(f" From ref1: {candidate_subsplits1}")
                print(f" From ref2: {candidate_subsplits2}")
            candidate_big_subsplits = chain.from_iterable(
                (big_subsplit for big_subsplit in subsplit1.cross(subsplit2))
                for subsplit1, subsplit2 in product(candidate_subsplits1, candidate_subsplits2)
            )
            for big_subsplit in candidate_big_subsplits:
                if stats is not None:
                    stats["potential_children_generated"] = stats.get("potential_children_generated", 0) + 1
                if not big_subsplit.is_strictly_valid():
                    continue
                if stats is not None:
                    stats["pcsp_generated"] = stats.get("pcsp_generated", 0) + 1
                pcsp = PCSP(parent_clade, big_subsplit)
                result.add(pcsp)
                if verbose:
                    print(f"  Adding {pcsp}")
                for child_clade in big_subsplit.nontrivial_children():
                    if stats is not None:
                        stats["childs_child_clades"] = stats.get("childs_child_clades", 0) + 1
                    if verbose:
                        print(f"   Considering child clade {child_clade}")
                    new_parent_subsplit_clade = SubsplitClade(big_subsplit, child_clade)
                    if verbose:
                        print(f"    New parent subsplit-clade: {new_parent_subsplit_clade}")

                    new_restricted_parent_subsplit_clade1 = new_parent_subsplit_clade.restrict(root_clade1)
                    if verbose:
                        print(f"    {new_parent_subsplit_clade} and {root_clade1} = {new_restricted_parent_subsplit_clade1}")
                    if not self.valid_parent(new_restricted_parent_subsplit_clade1, self_subsplit_set):
                        if verbose:
                            print(f"    {new_restricted_parent_subsplit_clade1} not valid parent, replacing with {restricted_parent_subsplit_clade1}")
                        new_restricted_parent_subsplit_clade1 = restricted_parent_subsplit_clade1
                    assert new_restricted_parent_subsplit_clade1.clade == child_clade & root_clade1, f"Clade mismatch: {new_restricted_parent_subsplit_clade1.clade} not equal to {child_clade & root_clade1}"

                    new_restricted_parent_subsplit_clade2 = new_parent_subsplit_clade.restrict(root_clade2)
                    if verbose:
                        print(f"    {new_parent_subsplit_clade} and {root_clade2} = {new_restricted_parent_subsplit_clade2}")
                    if not other.valid_parent(new_restricted_parent_subsplit_clade2, other_subsplit_set):
                        if verbose:
                            print(f"    {new_restricted_parent_subsplit_clade2} not valid parent, replacing with {restricted_parent_subsplit_clade2}")
                        new_restricted_parent_subsplit_clade2 = restricted_parent_subsplit_clade2
                    assert new_restricted_parent_subsplit_clade2.clade == child_clade & root_clade2, f"Clade mismatch: {new_restricted_parent_subsplit_clade2.clade} not equal to {child_clade & root_clade2}"

                    parent_clade_stack.append((new_parent_subsplit_clade, new_restricted_parent_subsplit_clade1, new_restricted_parent_subsplit_clade2))
                    if verbose:
                        print(f"   Stacking ({new_parent_subsplit_clade}, {new_restricted_parent_subsplit_clade1}, {new_restricted_parent_subsplit_clade2})")
        return result

    def add_tree(self, tree):
        # if isinstance(tree, MyTree):
        #     root_clade = tree.root_clade()
        # elif isinstance(tree, ete3.Tree):
        #     root_clade = Clade(tree.get_leaf_names())
        # else:
        #     raise TypeError("Argument 'tree' not ete3.Tree or MyTree.")
        for node in tree.traverse():
            pcsp = PCSP.from_node(node)
            if not pcsp.child.is_trivial():
                self.add(pcsp)

    def parent_clades(self):
        yield from iter(self.data.keys())

    def prune(self, verbose=False):
        result = self.copy()
        n = result._prune()
        if verbose:
            print(f"Number of pruning iterations: {n}")
        return result

    def _prune(self):
        n_iterations = 0
        did_anything = self._prune_once()
        while did_anything:
            did_anything = self._prune_once()
            n_iterations += 1
        return n_iterations

    def _prune_once(self):
        did_anything = False
        parent_clades = sorted(self.data.keys(), key=lambda x: len(x.clade))
        for parent_clade in parent_clades:
            if len(parent_clade.clade) < 2:
                continue
            children = list(self.data[parent_clade])
            for child in children:
                remove_child = False
                for child_clade in child.nontrivial_children():
                    if SubsplitClade(child, child_clade) not in self.data:
                        remove_child = True
                        did_anything = True
                if remove_child:
                    self.data[parent_clade].remove(child)
            if len(self.data[parent_clade]) == 0:
                self.data.pop(parent_clade)
                did_anything = True
        return did_anything

    @staticmethod
    def from_tree(tree):
        # if isinstance(tree, MyTree):
        #     tree = tree.tree
        result = PCSPSupport()
        result.add_tree(tree)
        return result

    @staticmethod
    def from_trees(trees):
        # if isinstance(tree, MyTree):
        #     tree = tree.tree
        result = PCSPSupport()
        for tree in trees:
            result.add_tree(tree)
        return result


class ProbabilityDistribution(collections.abc.MutableMapping):
    def __init__(self, arg=None, log=False, auto_normalize=False):
        self.params = dict()
        self.normalized = False
        self.auto_normalize = auto_normalize
        self.update(arg, log)

    @property
    def exp_params(self):
        return {key: math.exp(value) for (key, value) in self.params.items()}

    def __contains__(self, item):
        return item in self.params

    def __copy__(self):
        other = ProbabilityDistribution(self.params, log=True, auto_normalize=self.auto_normalize)
        other.normalized = self.normalized
        return other

    def __delitem__(self, key):
        self.remove(key)

    def __eq__(self, other):
        return self.items() == other.items()

    def __getitem__(self, item):
        return self.get(item, log=False)

    def __iter__(self):
        return iter(self.params.keys())

    def __len__(self):
        return len(self.params)

    def __repr__(self):
        return "ProbabilityDistribution("+repr(self.exp_params)+")"

    def __setitem__(self, item, probability):
        self.set(item, probability, log=False)

    def __str__(self):
        probs = self.exp_params
        strs = [str(key) + ": " + str(round(probs[key], 4)) for key in probs]
        return "{" + ", ".join(strs) + "}"

    def _log_sum_exp(self):
        params_np = np.array(list(self.params.values()))
        return np.log(np.sum(np.exp(params_np)))

    def _probs(self):
        return {key: math.exp(value) for (key, value) in self.params.items()}

    def _require_nonempty(self):
        if not self.params:
            raise ZeroDivisionError("No elements in ProbabilityDistribution.")

    def _require_normalization(self):
        if self.params and not self.normalized:
            if self.auto_normalize:
                self.normalize()
            else:
                raise RuntimeError("Distribution not normalized, use normalize() or set auto_normalize=True.")

    def add(self, item, value, log=False):
        if log:
            self.add_log(item, value)
        else:
            self.add_lin(item, value)

    def add_lin(self, item, prob):
        if prob == 0.0:
            return
        if item not in self:
            curr_prob = 0.0
        else:
            curr_prob = math.exp(self.params[item])
        self.set_lin(item, curr_prob + prob)

    def add_log(self, item, log_prob):
        if log_prob == 0.0:
            return
        if item in self:
            self.params[item] += log_prob
            self.normalized = False

    def add_many(self, arg, log=False, scalar=1.0):
        if log:
            self.add_many_log(arg, scalar)
        else:
            self.add_many_lin(arg, scalar)

    def add_many_lin(self, arg, scalar=1.0):
        for key in arg:
            self.add_lin(key, arg[key]*scalar)

    def add_many_log(self, arg, scalar=1.0):
        for key in arg:
            self.add_log(key, arg[key]*scalar)

    def check(self):
        return (not self.params) or self.normalized

    def copy(self):
        return self.__copy__()

    def degrees_of_freedom(self):
        return len(self) - 1

    def dust(self, threshold):
        result = self.copy()
        for item, prob in self.items():
            if prob < threshold:
                result.remove(item)
        if self.normalized:
            result.normalize()
        return result

    def get(self, item, log=False):
        if log:
            return self.get_log(item)
        else:
            return self.get_lin(item)

    def get_lin(self, item):
        if item not in self.params:
            return 0.0
        return math.exp(self.params.get(item))

    def get_log(self, item):
        if item not in self.params:
            return -math.inf
        return self.params.get(item)

    def items(self, log=False):
        # self._require_normalization()
        if log:
            return self.params.items()
        else:
            return self.exp_params.items()

    def keys(self):
        return self.params.keys()

    def likelihood(self, item):
        return self.prob_lin(item)

    def log_likelihood(self, item):
        return self.prob_log(item)

    def normalize(self):
        self._require_nonempty()
        lse = self._log_sum_exp()
        for key in self.params:
            self.params[key] -= lse
        self.normalized = True

    def prob(self, item, log=False):
        self._require_normalization()
        if log:
            return self.get_log(item)
        else:
            return self.get_lin(item)

    def prob_lin(self, item):
        return self.prob(item, log=False)

    def prob_log(self, item):
        return self.prob(item, log=True)

    def probs(self, log=False):
        if log:
            return self.probs_log()
        else:
            return self.probs_lin()

    def probs_lin(self):
        self._require_normalization()
        return dict(self.exp_params)

    def probs_log(self):
        self._require_normalization()
        return dict(self.params)

    def remove(self, item):
        if item in self.params:
            del self.params[item]
            self.normalized = False

    # def sample(self):
    #     items, probabilities = zip(*self.items())
    #     return random.choices(population=items, weights=probabilities)[0]

    def sample(self):
        return self.samples(k=1)[0]

    def samples(self, k=1):
        items, probabilities = zip(*self.items())
        return random.choices(population=items, weights=probabilities, k=k)

    def set(self, item, value, log=False):
        if log:
            self.set_log(item, value)
        else:
            self.set_lin(item, value)

    def set_lin(self, item, prob):
        if prob <= 0.0:
            self.remove(item)
        else:
            self.params[item] = math.log(prob)
            self.normalized = False

    def set_log(self, item, log_prob):
        if log_prob == -math.inf:
            self.remove(item)
        else:
            self.params[item] = log_prob
            self.normalized = False

    def set_many_lin(self, arg):
        if arg is None:
            return
        if not isinstance(arg, collections.abc.Mapping):
            print("Argument arg not a Mapping.")
            return
        for item in arg:
            self.set_lin(item, arg[item])

    def set_many_log(self, arg):
        if arg is None:
            return
        if not isinstance(arg, collections.abc.Mapping):
            print("Argument arg not a Mapping.")
            return
        for item in arg:
            self.set_log(item, arg[item])

    def set_many(self, arg=None, log=False):
        if arg is None:
            return
        if log:
            self.set_many_log(arg)
        else:
            self.set_many_lin(arg)

    def support(self):
        return set(self.params.keys())

    def truncate_below(self, cutoff, log=False):
        if not log:
            if cutoff <= 0.0:
                cutoff = -math.inf
            else:
                cutoff = math.log(cutoff)
        if cutoff == 0.0:
            return
        for key in list(self.keys()):
            if self.params[key] < cutoff:
                self.remove(key)
                self.normalized = False

    def update(self, arg=None, log=False):
        self.set_many(arg, log)

    def max_item(self):
        return max(self.params.items(), key=itemgetter(1))[0]

    def max_likelihood(self):
        self._require_normalization()
        return max(self.exp_params.values())

    def max_log_likelihood(self):
        self._require_normalization()
        return max(self.params.values())

    def kl_divergence(self, other):
        result = 0.0
        for item in self:
            log_p = self.prob_log(item)
            log_q = other.prob_log(item)
            p = self.prob_lin(item)
            if p > 0.0:
                result += -p*(log_q - log_p)
        return result

    @staticmethod
    def random(support, concentration=1.0, cutoff=0.0):
        distribution = np.random.dirichlet([concentration] * len(support))
        pr = ProbabilityDistribution({item: prob for item, prob in zip(support, distribution)})
        pr.truncate_below(cutoff)
        pr.normalize()
        return pr

    @staticmethod
    def random_with_sparsity(items, sparsity, concentration=1.0):
        n_realized_items = binomial_min_one(len(items), 1 - sparsity)
        support = random.sample(items, n_realized_items)
        pr = ProbabilityDistribution.random(support, concentration)
        return pr

    @staticmethod
    def from_list(items: list):
        count = Counter(items)
        n = len(items)
        return ProbabilityDistribution({item: k/n for item, k in count.items()})


class TreeDistribution(ProbabilityDistribution):
    def __repr__(self):
        return "TreeDistribution(" + repr(self.exp_params) + ")"

    def __str__(self):
        string_queue = []
        probs = self.exp_params
        for tree in self:
            string_queue.append(str(tree))
            string_queue.append(f"{probs[tree]:0.4}")
        return "\n".join(string_queue)

    @staticmethod
    def _refresh_attributes(tree):
        for node in tree.traverse("postorder"):
            clade = Clade(node.get_leaf_names())
            node.clade = clade
            if node.is_leaf():
                node.name = str(clade)
            else:
                child_clades = [child.clade for child in node.children]
                assert len(child_clades) == 2
                subsplit = Subsplit(*child_clades)
                node.subsplit = subsplit
                node.name = str(subsplit)

    # Lots of MyTree alterations
    def restrict(self, taxon_set):
        result = TreeDistribution()
        # id_to_tree = {}
        for tree in self:
            restricted_tree = tree.prune(taxon_set)
            # self._refresh_attributes(tree)
            result.add(restricted_tree, self.get(tree))
            # tree_id = tree.get_topology_id()
            # if tree_id not in id_to_tree:
            #     result[tree_copy] = self.get(tree)
            #     id_to_tree[tree_id] = tree_copy
            # else:
            #     matching_tree = id_to_tree[tree_id]
            #     result[matching_tree] += self.get(tree)
        result.normalize()
        return result

    def feature_prob(self, item):
        result = 0.0
        for tree in self:
            if item in tree:
                result += self[tree]
        return result

    def prob_all(self, items):
        result = 0.0
        for tree in self:
            if all(item in tree for item in items):
                result += self[tree]
        return result

    def prob_any(self, items):
        result = 0.0
        for tree in self:
            if any(item in tree for item in items):
                result += self[tree]
        return result

    @staticmethod
    def from_list(items: Sequence[MyTree]):
        count = Counter(items)
        n = len(items)
        return TreeDistribution({item: k / n for item, k in count.items()})


class SBN:
    def __init__(self, cond_probs=None):
        self.data = dict()
        self.update(cond_probs)

    def __copy__(self):
        result = SBN()
        for parent in self.parents():
            result.data[parent] = self.data[parent].copy()
        return result

    def __eq__(self, other):
        if not isinstance(other, SBN):
            return False
        # if not self.parents() == other.parents():
        #     return False
        # for parent in self.parents():
        #     if not self[parent] == other[parent]:
        #         return False
        return self.data == other.data

    def __repr__(self):
        pretty_repr = pprint.pformat(dict(self.data))
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join(("SBN(", pretty_repr, ")"))

    def __str__(self):
        strs = [str(parent) + ": " + str(self.get(parent)) for parent in sorted(self.parents())]
        return "\n".join(strs)

    def __getitem__(self, item):
        if isinstance(item, Subsplit):
            parent_clade1 = SubsplitClade(item, item.clade1)
            parent_clade2 = SubsplitClade(item, item.clade2)
            return self.data[parent_clade1], self.data[parent_clade2]
        if isinstance(item, SubsplitClade):
            return self.data[item]
        if isinstance(item, PCSP):
            parent_clade = item.parent_clade()
            child = item.child
            return self.data[parent_clade][child]

    def __len__(self):
        result = 0
        for parent in self.parents():
            result += len(self[parent])
        return result

    def add(self, pcsp, value, log=False):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        if log:
            self.add_log(pcsp, value)
        else:
            self.add_lin(pcsp, value)

    def add_lin(self, pcsp, prob):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        if prob == 0.0:
            return
        parent = pcsp.parent_clade()
        child = pcsp.child
        if parent not in self.data:
            self.data[parent] = ProbabilityDistribution()
        self.data[parent].add_lin(child, prob)

    def add_log(self, pcsp, log_prob):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        if log_prob == 0.0:
            return
        parent = pcsp.parent_clade()
        child = pcsp.child
        if parent not in self.data:
            # self.data[clade] = ProbabilityDistribution()
            return
        self.data[parent].add_log(child, log_prob)

    def add_many(self, arg, log=False, scalar=1.0):
        if log:
            self.add_many_log(arg, scalar)
        else:
            self.add_many_lin(arg, scalar)

    def add_many_lin(self, arg, scalar=1.0):
        for key in arg:
            self.add_lin(key, arg[key]*scalar)

    def add_many_log(self, arg, scalar=1.0):
        for key in arg:
            self.add_log(key, arg[key]*scalar)

    def bad_distributions(self):
        result = set()
        for parent in self.data:
            if not self.data[parent].check():
                result.add(parent)
        return result

    def bitarray_summary(self, bit_map=None):
        if bit_map is None:
            bit_map = sorted(self.root_clade())
        n = len(bit_map)
        result = dict()
        for pcsp in self.iter_pcsp():
            assert pcsp not in result, "Visited same PCSP more than once."
            bits = pcsp.to_bitarray(bit_map)
            if pcsp.parent == self.root_subsplit():
                key = bits[2*n:].to01()
            else:
                key = "|".join(bits[i*n:(i+1)*n].to01() for i in range(3))
            result[key] = self.get(pcsp)
        return result

    def check(self):
        return self.check_distributions() and self.support().is_complete()

    def check_distributions(self):
        for parent in self.data:
            if not self.data[parent].check():
                return False
        return True

    def copy(self):
        return self.__copy__()

    def degrees_of_freedom(self):
        result = 0
        for parent in self.parents():
            result += self[parent].degrees_of_freedom()
        return result

    def equiv_classes(self, restriction, include_root=True):
        result = dict()
        if include_root:
            result[Subsplit(restriction)] = {self.root_subsplit()}
        for subsplit in self.iter_subsplits(include_root=False):
            subsplit_res = subsplit.restrict(restriction)
            if subsplit_res.is_trivial():
                continue
            if subsplit_res not in result:
                result[subsplit_res] = set()
            result[subsplit_res].add(subsplit)
        return result

    def get(self, item):
        if isinstance(item, Subsplit):
            parent_clade1 = SubsplitClade(item, item.clade1)
            parent_clade2 = SubsplitClade(item, item.clade2)
            return self.data.get(parent_clade1, ProbabilityDistribution()), \
                self.data.get(parent_clade2, ProbabilityDistribution())
        if isinstance(item, SubsplitClade):
            return self.data.get(item, ProbabilityDistribution())
        if isinstance(item, PCSP):
            parent = item.parent_clade()
            child = item.child
            return self.data.get(parent, ProbabilityDistribution()).get(child)

    def get_log(self, item):
        if not isinstance(item, PCSP):
            raise TypeError("Argument 'item' not of class PCSP.")
        parent_clade = item.parent_clade()
        return self[parent_clade].get_log(item.child)

    def gradient_descent(self, references: List['SBN'], starting_gamma=1.0, true_reference: 'SBN' = None,
                         max_iteration=50, verbose=True, const=0.5, shrink_factor=0.5, grow_factor=1.1):
        weights = [1] * len(references)
        restrictions = [reference.root_clade() for reference in references]
        current = self.copy()
        current_rs = [current.restrict(restriction) for restriction in restrictions]
        current_kl = sum(weight * reference.kl_divergence(current_r) for (weight, reference, current_r) in
                         zip(weights, references, current_rs))
        current_transit = current.transit_probabilities()
        current_grad = current.restricted_kl_gradient_multi(others=references, weights=weights,
                                                            transit=current_transit, restricted_selves=current_rs)
        current_grad_l2 = l2norm2(current_grad)
        gamma = starting_gamma
        kl_list = [current_kl]
        if true_reference is not None:
            true_kl_list = [true_reference.kl_divergence(current)]
        iteration = 0
        if verbose:
            print(f"Iter {iteration}: KL={current_kl:8.4g}")
        while iteration < max_iteration:
            iteration += 1
            candidate = current.copy()
            candidate.add_many_log(current_grad, -gamma)
            candidate.normalize()
            candidate_rs = [candidate.restrict(restriction) for restriction in restrictions]
            candidate_kl = sum(weight * reference.kl_divergence(candidate_r) for (weight, reference, candidate_r) in
                               zip(weights, references, candidate_rs))
            if verbose:
                print(f"Iter {iteration}: KL={candidate_kl:8.4g}")
            # Armijo-Goldstein condition:
            # see https://math.stackexchange.com/questions/373868/optimal-step-size-in-gradient-descent
            if candidate_kl <= current_kl - const * gamma * current_grad_l2:
                gamma = gamma * grow_factor
            else:
                gamma = gamma * shrink_factor
                if verbose:
                    print(f"Shrinking gamma to {gamma}")
            if candidate_kl <= current_kl:
                current = candidate
                current_rs = candidate_rs
                current_kl = candidate_kl
                current_transit = current.transit_probabilities()
                current_grad = current.restricted_kl_gradient_multi(others=references, weights=weights,
                                                                    transit=current_transit,
                                                                    restricted_selves=current_rs)
                current_grad_l2 = l2norm2(current_grad)
                # if verbose:
                #     print("Move accepted")
            else:
                if verbose:
                    print("Move rejected!")
            kl_list.append(current_kl)
            if true_reference is not None:
                true_kl_list.append(true_reference.kl_divergence(current))
        if verbose:
            print(f"Final KL={current_kl:8.4g}")
        if true_reference is not None:
            return current, kl_list, true_kl_list
        return current, kl_list

    # TODO: consider implementing inline so as not to have to copy everything to a Support object
    def is_complete(self):
        return self.support().is_complete()

    # TODO: Compare usefulness to parents()
    def iter_parents(self):
        yield from self.parents()

    # TODO: check for usefulness
    def iter_clades(self):
        clade_stack = [self.root_clade()]
        visited_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in visited_clades:
                continue
            visited_clades.add(clade)
            yield clade

    def iter_subsplits(self, include_root: bool = False):
        visited_subsplits = set()
        if include_root:
            root_subsplit = self.root_subsplit()
            visited_subsplits.add(root_subsplit)
            yield root_subsplit
        for parent_clade in self.data:
            for subsplit in self.data[parent_clade]:
                if subsplit in visited_subsplits:
                    continue
                visited_subsplits.add(subsplit)
                yield subsplit

    # TODO: Update
    # def iter_probabilities(self):
    #     clade_stack = [(self.root_clade(), 1.0)]
    #     while clade_stack:
    #         clade, clade_prob = clade_stack.pop()
    #         cond_dist = self[clade]
    #         for subsplit in cond_dist:
    #             cond_prob = self[subsplit]
    #             joint_prob = clade_prob * cond_prob
    #             yield subsplit, cond_prob, joint_prob
    #             for child in subsplit.nontrivial_children():
    #                 clade_stack.append((child, joint_prob))

    def iter_pcsp(self):
        for parent_clade in self.iter_parents():
            for child in self.data[parent_clade]:
                yield PCSP(parent_clade.subsplit, child)

    def prob_lin(self, tree):
        return self.likelihood(tree)

    def prob_log(self, tree):
        return self.log_likelihood(tree)

    def probabilities(self):
        parents = sorted(self.parents(), key=lambda par: (-len(par.subsplit), par.subsplit.clade(), -len(par.clade)))
        subsplit_probs = dict()
        pcsp_probs = dict()
        subsplit_probs[self.root_subsplit()] = 1.0
        for parent in parents:
            for child in self[parent]:
                pcsp = PCSP(parent, child)
                uncond_prob = subsplit_probs[parent.subsplit] * self.get(pcsp)
                if pcsp not in pcsp_probs:
                    pcsp_probs[pcsp] = 0.0
                pcsp_probs[pcsp] += uncond_prob
                if child not in subsplit_probs:
                    subsplit_probs[child] = 0.0
                subsplit_probs[child] += uncond_prob
        return subsplit_probs, pcsp_probs

    def pcsp_probabilities(self, verbose=False):
        parents = sorted(self.parents(), key=lambda par: (-len(par.subsplit), par.subsplit.clade(), -len(par.clade)))
        parent_probs = dict()
        result = dict()
        parent_probs[self.root_subsplit()] = 1.0
        for parent in parents:
            for child in self[parent]:
                pcsp = PCSP(parent, child)
                parent_prob = parent_probs.get(parent.subsplit, 0.0)
                if parent_prob == 0.0 and verbose:
                    print(f"Parent with zero probability: {parent.subsplit}")
                uncond_prob = parent_prob * self.get(pcsp)
                if pcsp not in result:
                    result[pcsp] = 0.0
                result[pcsp] += uncond_prob
                if child not in parent_probs:
                    parent_probs[child] = 0.0
                parent_probs[child] += uncond_prob
        return result

    # TODO: Update
    # def iter_unconditional_probabilities(self):
    #     clade_stack = [(self.root_clade(), 1.0)]
    #     while clade_stack:
    #         clade, clade_prob = clade_stack.pop()
    #         cond_dist = self[clade]
    #         for subsplit in cond_dist:
    #             cond_prob = self[subsplit]
    #             joint_prob = clade_prob * cond_prob
    #             yield subsplit, joint_prob
    #             for child in subsplit.nontrivial_children():
    #                 clade_stack.append((child, joint_prob))

    def clade_probabilities(self):
        subsplit_probs = self.subsplit_probabilities()
        result = dict()
        for subsplit, prob in subsplit_probs.items():
            for child_clade in subsplit.nontrivial_children():
                if child_clade not in result:
                    result[child_clade] = 0.0
                result[child_clade] += prob
        return result

    def subsplit_probabilities(self):
        parent_clades = sorted(
            self.parents(),
            key=lambda par: (
                -len(par.subsplit),
                par.subsplit.clade(),
                -len(par.clade)
            )
        )
        result = dict()
        result[self.root_subsplit()] = 1.0
        for parent_clade in parent_clades:
            for child in self[parent_clade]:
                uncond_prob = (
                        result[parent_clade.subsplit]
                        * self.get(PCSP(parent_clade, child))
                )
                if child not in result:
                    result[child] = 0.0
                result[child] += uncond_prob
        return result

    def transit_probabilities(self):
        parent_clades = sorted(
            self.parents(),
            key=lambda par: (
                -len(par.subsplit),
                par.subsplit.clade(),
                -len(par.clade)
            )
        )
        result = {self.root_subsplit(): {self.root_subsplit(): 1.0}}
        for parent_clade in parent_clades:
            parent = parent_clade.subsplit
            # clade = parent_clade.clade
            # if parent not in result:
            #     result[parent] = dict()
            # result[parent][parent] = 1.0
            curr_paths = result[parent].copy()
            curr_paths[parent_clade] = 1.0
            for child in self[parent_clade]:
                pcsp = PCSP(parent, child)
                cond_prob = self[pcsp]
                if child not in result:
                    result[child] = {child: 1.0}
                # result[child][parent_clade] = cond_prob
                for ancestor in curr_paths:
                    path_prob = curr_paths[ancestor] * cond_prob
                    # if child not in result:
                    #     result[child] = dict()
                    if ancestor not in result[child]:
                        result[child][ancestor] = 0.0
                    result[child][ancestor] += path_prob
        return result

    def kl_divergence_scd(self, other):
        result = 0.0
        pcsp_probs = self.pcsp_probabilities()
        for pcsp in pcsp_probs:
            result += -pcsp_probs[pcsp] * (other.get_log(pcsp) - self.get_log(pcsp))
        return result

    def kl_divergence_scd_verbose(self, other):
        result = {}
        pcsp_probs = self.pcsp_probabilities()
        for pcsp in pcsp_probs:
            if pcsp.parent not in result:
                result[pcsp.parent] = 0.0
            result[pcsp.parent] += -pcsp_probs[pcsp] * (other.get_log(pcsp) - self.get_log(pcsp))
        return result

    def kl_divergence_treedist(self, other):
        result = 0.0
        for tree in other:
            log_q = other.log_likelihood(tree)
            log_p = self.log_likelihood(tree)
            result += -math.exp(log_p) * (log_q - log_p)
        return result

    def kl_divergence(self, other):
        if isinstance(other, SBN):
            return self.kl_divergence_scd(other)
        if isinstance(other, TreeDistribution):
            return self.kl_divergence_treedist(other)
        raise TypeError("Argument 'other' not a SBN or TreeDistribution")

    def log_likelihood(self, tree, strategy='levelorder'):
        result = 0.0
        for pcsp in tree.traverse_pcsps(strategy):
            cond_prob = self[pcsp]
            if cond_prob > 0.0:
                result += math.log(cond_prob)
            else:
                result += -math.inf
        return result

    def likelihood(self, tree, strategy='levelorder'):
        return math.exp(self.log_likelihood(tree, strategy))

    # TODO: parallel to max_clade_tree
    def max_clade_tree_draft(self):
        clade_probs = self.clade_probabilities()
        clades = sorted(clade_probs.keys(), key=lambda cl: (len(cl), cl))
        best_subtree_and_prob = {}
        for clade in clades:
            best_subsplit = None
            best_subtree_prob = 0.0
            clade_score = clade_probs[clade]
            for subsplit in self[clade]:
                cond_prob = clade_score
                subtree_prob = cond_prob
                for child_clade in subsplit.nontrivial_children():
                    subtree_prob *= best_subtree_and_prob[child_clade][1]
                if subtree_prob > best_subtree_prob:
                    best_subsplit = subsplit
                    best_subtree_prob = subtree_prob
            best_subtree = SubsplitSupport(best_subsplit)
            for child_clade in best_subsplit.nontrivial_children():
                best_subtree.update(best_subtree_and_prob[child_clade][0])
            best_subtree_and_prob[clade] = (best_subtree, best_subtree_prob)
        best_tree_support, best_prob = best_subtree_and_prob[self.root_clade()]
        return best_tree_support.random_tree(), best_prob

    def max_clade_tree(self):
        parent_clades = sorted(self.parents(), key=lambda p_c: (len(p_c.clade), p_c.clade))
        clade_probs = self.clade_probabilities()
        best_subtree_and_score = {}
        for parent_clade in parent_clades:
            best_subsplit = None
            best_subtree_score = 0.0
            clade_score = clade_probs[parent_clade.clade]
            for subsplit in self[parent_clade]:
                subtree_score = clade_score
                for child_clade in subsplit.nontrivial_children():
                    child_subsplit_clade = SubsplitClade(subsplit, child_clade)
                    subtree_score *= best_subtree_and_score[child_subsplit_clade][1]
                if subtree_score > best_subtree_score:
                    best_subsplit = subsplit
                    best_subtree_score = subtree_score
            best_pcsp = PCSP(parent_clade, best_subsplit)
            best_subtree = PCSPSupport(best_pcsp)
            for child_clade in best_subsplit.nontrivial_children():
                child_subsplit_clade = SubsplitClade(best_subsplit, child_clade)
                best_subtree.update(best_subtree_and_score[child_subsplit_clade][0])
            best_subtree_and_score[parent_clade] = (best_subtree, best_subtree_score)
        best_tree_support, best_score = best_subtree_and_score[self.root_subsplit_clade()]
        return best_tree_support.random_tree(), best_score

    def max_prob_tree(self):
        parent_clades = sorted(self.parents(), key=lambda p_c: (len(p_c.clade), p_c.clade))
        best_subtree_and_prob = {}
        for parent_clade in parent_clades:
            best_subsplit = None
            best_subtree_prob = 0.0
            for subsplit, cond_prob in self[parent_clade].items():
                # cond_prob = self[subsplit]
                subtree_prob = cond_prob
                for child_clade in subsplit.nontrivial_children():
                    child_subsplit_clade = SubsplitClade(subsplit, child_clade)
                    subtree_prob *= best_subtree_and_prob[child_subsplit_clade][1]
                if subtree_prob > best_subtree_prob:
                    best_subsplit = subsplit
                    best_subtree_prob = subtree_prob
            best_pcsp = PCSP(parent_clade, best_subsplit)
            best_subtree = PCSPSupport(best_pcsp)
            for child_clade in best_subsplit.nontrivial_children():
                child_subsplit_clade = SubsplitClade(best_subsplit, child_clade)
                best_subtree.update(best_subtree_and_prob[child_subsplit_clade][0])
            best_subtree_and_prob[parent_clade] = (best_subtree, best_subtree_prob)
        best_tree_support, best_prob = best_subtree_and_prob[self.root_subsplit_clade()]
        return best_tree_support.random_tree(), best_prob

    def normalize(self, parent=None):
        if parent is None:
            for parent in self.data:
                self.data[parent].normalize()
        elif isinstance(parent, SubsplitClade):
            self.data[parent].normalize()
        elif isinstance(parent, Subsplit):
            for child_clade in parent.nontrivial_children():
                parent_clade = SubsplitClade(parent, child_clade)
                self.data[parent_clade].normalize()
        elif isinstance(parent, Clade) and parent == self.root_clade():
            self.data[self.root_subsplit_clade()].normalize()
        else:
            raise TypeError("Argument 'parent' not None, SubsplitClade, Subsplit, or root Clade")

    def parents(self):
        return self.data.keys()

    def prune(self, verbose=False):
        result = self.copy()
        n = result._prune()
        if verbose:
            print(f"Number of pruning iterations: {n}")
        return result

    def _prune(self):
        n_iterations = 0
        did_anything = self._prune_once()
        while did_anything:
            did_anything = self._prune_once()
            n_iterations += 1
        return n_iterations

    def _prune_once(self):
        did_anything = False
        parent_clades = sorted(self.data.keys(), key=lambda x: len(x.clade))
        for parent_clade in parent_clades:
            if len(parent_clade.clade) < 2:
                continue
            children = list(self.data[parent_clade])
            for child in children:
                remove_child = False
                for child_clade in child.nontrivial_children():
                    if SubsplitClade(child, child_clade) not in self.data:
                        remove_child = True
                        did_anything = True
                if remove_child:
                    self.data[parent_clade].remove(child)
            if len(self.data[parent_clade]) == 0:
                self.data.pop(parent_clade)
                did_anything = True
        return did_anything

    def random_tree(self):
        # if not self.is_complete():
        #     raise ValueError("SBN not complete, no tree possible.")
        root_subsplit_clade = self.root_subsplit_clade()
        tree = MyTree()

        parent_clade_stack = [(tree.tree, root_subsplit_clade)]  # MyTree alteration
        while parent_clade_stack:
            node, parent_clade = parent_clade_stack.pop()
            subsplit = self.data[parent_clade].sample()
            node.name = str(subsplit)
            # assert len(subsplit) == 2
            left = subsplit.clade1
            right = subsplit.clade2
            assert len(left) > 0 and len(right) > 0
            if len(left) > 1:
                left_node = node.add_child()
                subsplit_clade = SubsplitClade(subsplit, left)
                parent_clade_stack.append((left_node, subsplit_clade))
            else:
                node.add_child(name=str(left))
            if len(right) > 1:
                right_node = node.add_child()
                subsplit_clade = SubsplitClade(subsplit, right)
                parent_clade_stack.append((right_node, subsplit_clade))
            else:
                node.add_child(name=str(right))
        return tree

    def remove(self, pcsp: PCSP):
        parent_clade = pcsp.parent_clade()
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        self.data[parent_clade].remove(pcsp.child)

    # TODO: Add correct handling if arg is not iterable
    def remove_many(self, arg):
        for pcsp in arg:
            self.remove(pcsp)

    def restrict(self, taxon_set, verbose=False):
        taxon_set = Clade(taxon_set)
        root_subsplit = self.root_subsplit()
        # restricted_root_subsplit = root_subsplit.restrict(taxon_set)
        root_subsplit_clade = self.root_subsplit_clade()
        result = SBN()
        t_probabilities = self.transit_probabilities()
        for child in t_probabilities:
            restricted_child = child.restrict(taxon_set)
            if verbose:
                print(f"Considering child {child} with restriction {restricted_child}.")
            if restricted_child.is_trivial():
                if verbose:
                    print(f"Restricted child {restricted_child} is trivial, skipping.")
                continue
            for ancestor in t_probabilities[child]:
                if not isinstance(ancestor, SubsplitClade):
                    continue
                restricted_ancestor = ancestor.restrict(taxon_set)
                if verbose:
                    print(f"Considering ancestor {ancestor} with restriction {restricted_ancestor}.")
                if (restricted_ancestor.subsplit.is_trivial() and ancestor != root_subsplit_clade) or \
                        not restricted_ancestor.subsplit.valid_child(restricted_child):
                    if verbose:
                        print(f"Restricted ancestor {restricted_ancestor} is trivial and not root "
                              f"or incompatible with {restricted_child}, skipping.")
                    continue
                prob1 = t_probabilities[ancestor.subsplit][root_subsplit]
                prob2 = t_probabilities[child][ancestor]
                pcsp_prob = prob1 * prob2
                result.add(PCSP(restricted_ancestor, restricted_child), pcsp_prob)
        result.normalize()
        return result

    def restrict_transit(self, restriction, transit=None):
        root_subsplit = self.root_subsplit()
        if transit is None:
            transit = self.transit_probabilities()
        result = dict()
        for destination in transit:
            dest_res = destination.restrict(restriction)
            if dest_res.is_trivial() and destination != root_subsplit:
                continue
            if dest_res not in result:
                result[dest_res] = dict()
            for ancestor in transit[destination]:
                if ancestor not in result[dest_res]:
                    result[dest_res][ancestor] = 0.0
                result[dest_res][ancestor] += transit[destination][ancestor]
        return result

    def root_clade(self):
        return Clade(self.taxon_set())

    def root_subsplit(self):
        return Subsplit(self.root_clade())

    def root_subsplit_clade(self):
        root_clade = self.root_clade()
        return SubsplitClade(Subsplit(root_clade), Clade(root_clade))

    def set(self, pcsp, value, log=False):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        if log:
            self.set_log(pcsp, value)
        else:
            self.set_lin(pcsp, value)

    def set_lin(self, pcsp, prob):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        parent_clade = pcsp.parent_clade()
        if prob > 0.0 and parent_clade not in self.data:
            self.data[parent_clade] = ProbabilityDistribution()
        self.data[parent_clade].set_lin(pcsp.child, prob)

    def set_log(self, pcsp, log_prob):
        if not isinstance(pcsp, PCSP):
            raise TypeError("Argument 'pcsp' not class PCSP")
        parent_clade = pcsp.parent_clade()
        if log_prob > -math.inf and parent_clade not in self.data:
            self.data[parent_clade] = ProbabilityDistribution()
        self.data[parent_clade].set_log(pcsp.child, log_prob)

    def set_many(self, arg, log=False):
        if log:
            self.set_many_log(arg)
        else:
            self.set_many_lin(arg)

    def set_many_lin(self, arg):
        if arg is None:
            return
        for key in arg:
            self.set_lin(key, arg[key])

    def set_many_log(self, arg):
        if arg is None:
            return
        for key in arg:
            self.set_log(key, arg[key])

    # TODO: Verify
    def support(self):
        return PCSPSupport(self.iter_pcsp())

    def taxon_set(self):
        return set().union(*(parent.subsplit.clade() for parent in self.data))

    def tree_distribution(self, verbose=False):
        # if not self.is_complete():
        #     raise TypeError("SBN not complete, no tree possible.")
        root_clade = self.root_clade()
        root_subsplit = self.root_subsplit()
        tree = MyTree()
        tree.tree.clade = root_clade
        tree.tree.parent_subsplit = root_subsplit
        result = TreeDistribution()
        big_stack = [(tree.tree, 0.0)]
        while big_stack:
            current_tree, log_prob_so_far = big_stack.pop()
            if verbose:
                print(f"Current tree: {current_tree}")
                print(f"Current log-probability: {log_prob_so_far}")
            possibilities = []
            state_stack = [(leaf.clade, leaf.parent_subsplit)
                           for leaf in current_tree.iter_leaves()
                           if len(leaf.clade) > 1]
            if not state_stack:
                result.set_log(MyTree(current_tree), log_prob_so_far)
                if verbose:
                    print(f"Sending to output: {current_tree} with log-probability {log_prob_so_far}")
                continue
            for clade, parent_subsplit in state_stack:
                parent_clade = SubsplitClade(parent_subsplit, clade)
                if verbose:
                    print(f"Examining clade: {clade} from parent {parent_subsplit}")
                possibilities.append(subsplit for subsplit, prob in self.data[parent_clade].items()
                                     if prob > 0.0 and not subsplit.is_trivial())
            for new_subsplits in product(*possibilities):
                if verbose:
                    print(f"Examining possibility: {new_subsplits}")
                new_tree = current_tree.copy()
                new_log_prob = log_prob_so_far
                for subsplit in new_subsplits:
                    node = next(new_tree.iter_search_nodes(clade=subsplit.clade()))
                    node.subsplit = subsplit
                    node.name = str(subsplit)
                    parent = node.parent_subsplit
                    pcsp = PCSP(parent, subsplit)
                    new_log_prob += self.get_log(pcsp)
                    for child_clade in subsplit.children():
                        child_node = node.add_child(name=str(child_clade))
                        child_node.clade = child_clade
                        child_node.parent_subsplit = subsplit
                        if verbose:
                            print(f"Pushing clade: {child_clade} with parent {subsplit}")
                big_stack.append((new_tree, new_log_prob))
                if verbose:
                    print(f"Pushing tree: {new_tree} with probability {math.exp(new_log_prob)}")
        result.normalize()
        return result

    # TODO: Update
    # def unconditional_probabilities(self):
    #     result = defaultdict(float)
    #     clade_stack = [(self.root_clade(), 1.0)]
    #     while clade_stack:
    #         clade, clade_prob = clade_stack.pop()
    #         cond_dist = self[clade]
    #         for subsplit in cond_dist:
    #             cond_prob = self[subsplit]
    #             joint_prob = clade_prob * cond_prob
    #             result[subsplit] += joint_prob
    #             for child in subsplit.nontrivial_children():
    #                 clade_stack.append((child, joint_prob))
    #     return result

    def update(self, arg, log=False):
        self.set_many(arg, log)

    # def update_old(self, parent, cond_probs):
    #     if cond_probs is None:
    #         return
    #     if not isinstance(parent, Subsplit):
    #         raise TypeError("Argument 'parent' not class Subsplit")
    #     if not isinstance(cond_probs, collections.abc.Mapping):
    #         raise TypeError("Argument 'cond_probs' not a Mapping")
    #     for subsplit in cond_probs:
    #         if not isinstance(subsplit, Subsplit):
    #             print("Warning: Non-Subsplit found in cond_probs, skipping")
    #             continue
    #         self.data[parent][subsplit] = cond_probs[subsplit]

    def subsplit_derivative(self, prob_of: Subsplit, wrt: PCSP, transit: dict = None):
        root_subsplit = self.root_subsplit()
        wrt_parent = wrt.parent
        wrt_parent_clade = wrt.parent_clade()
        wrt_child = wrt.child
        if transit is None:
            transit = self.transit_probabilities()
        uncond_wrt = transit.get(wrt_parent, dict()).get(root_subsplit, 0.0) * self[wrt]
        to_prob_of = transit.get(prob_of, dict())
        child_to_prob_of = to_prob_of.get(wrt_child, 0.0)
        parent_to_prob_of = to_prob_of.get(wrt_parent_clade, 0.0)
        return uncond_wrt * (child_to_prob_of - parent_to_prob_of)

    def subsplit_to_subsplit_cond_derivative(self, prob_of: Subsplit, cond_on: Subsplit, wrt: PCSP,
                                             transit: dict = None):
        parent = wrt.parent
        parent_clade = wrt.parent_clade()
        child = wrt.child
        if transit is None:
            transit = self.transit_probabilities()
        uncond_wrt = transit.get(parent, dict()).get(cond_on, 0.0) * self[wrt]
        to_prob_of = transit.get(prob_of, dict())
        child_to_prob_of = to_prob_of.get(child, 0.0)
        parent_to_prob_of = to_prob_of.get(parent_clade, 0.0)
        return uncond_wrt * (child_to_prob_of - parent_to_prob_of)

    def subsplit_via_subsplit_derivative(self, prob_of: Subsplit, via: Subsplit, wrt: PCSP, transit: dict = None):
        root_subsplit = self.root_subsplit()
        if transit is None:
            transit = self.transit_probabilities()
        da_b = (self.subsplit_to_subsplit_cond_derivative(prob_of=via, cond_on=root_subsplit, wrt=wrt, transit=transit)
                * transit.get(prob_of, dict()).get(via, 0.0))
        a_db = (transit.get(via, dict()).get(root_subsplit, 0.0)
                * self.subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=via, wrt=wrt, transit=transit))
        return da_b + a_db

    def restricted_subsplit_derivative(self, restriction: Clade, prob_of: Subsplit, wrt: PCSP, transit: dict = None):
        root_subsplit = self.root_subsplit()
        if transit is None:
            transit = self.transit_probabilities()
        result = 0.0
        for subsplit in self.iter_subsplits(include_root=True):
            restricted_subsplit = subsplit.restrict(restriction)
            if not restricted_subsplit == prob_of or (restricted_subsplit.is_trivial() and subsplit != root_subsplit):
                continue
            result += self.subsplit_derivative(prob_of=subsplit, wrt=wrt, transit=transit)
        return result

    def restricted_pcsp_derivative(self, restriction: Clade, prob_of: PCSP, wrt: PCSP, transit: dict = None):
        root_subsplit = self.root_subsplit()
        parent = prob_of.parent
        child = prob_of.child
        if transit is None:
            transit = self.transit_probabilities()
        result = 0.0
        for destination in transit:
            dest_res = destination.restrict(restriction)
            if dest_res != child:
                continue
            for ancestor in transit[destination]:
                if not isinstance(ancestor, Subsplit):
                    continue
                ansr_res = ancestor.restrict(restriction)
                if ansr_res != parent or (ansr_res.is_trivial() and ancestor != root_subsplit):
                    continue
                result += self.subsplit_via_subsplit_derivative(prob_of=destination, via=ancestor, wrt=wrt,
                                                                transit=transit)
        return result

    def restricted_conditional_derivative(self, restriction: Clade, prob_of: PCSP, wrt: PCSP,
                                          transit: dict = None, restricted_scd: 'SBN' = None,
                                          restricted_subsplit_probs: dict = None,
                                          restricted_pcsp_probs: dict = None):
        if transit is None:
            transit = self.transit_probabilities()
        if restricted_subsplit_probs is None or restricted_pcsp_probs is None:
            if restricted_scd is None:
                restricted_scd = self.restrict(restriction)
            if restricted_subsplit_probs is None:
                restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
            if restricted_pcsp_probs is None:
                restricted_pcsp_probs = restricted_scd.pcsp_probabilities()
        # Quotient rule d(top/bot) = (bot*dtop-top*dbot) / bot**2
        top = restricted_pcsp_probs[prob_of]
        bot = restricted_subsplit_probs[prob_of.parent]
        dtop = self.restricted_pcsp_derivative(
            restriction=restriction, prob_of=prob_of, wrt=wrt, transit=transit
        )
        dbot = self.restricted_subsplit_derivative(
            restriction=restriction, prob_of=prob_of.parent,
            wrt=wrt, transit=transit
        )
        return (bot * dtop - top * dbot) / bot ** 2

    def restricted_kl_derivative(self, wrt: PCSP, other: 'SBN', transit: dict = None,
                                 restricted_scd: 'SBN' = None, other_pcsp_probs: dict = None):
        restriction = other.root_clade()
        if not restriction.issubset(self.root_clade()):
            raise ValueError("Non-concentric taxon sets.")
        if transit is None:
            transit = self.transit_probabilities()
        if restricted_scd is None:
            restricted_scd = self.restrict(restriction)
        if other_pcsp_probs is None:
            other_pcsp_probs = other.pcsp_probabilities()
        result = 0.0
        for other_pcsp in other.iter_pcsp():
            other_pcsp_prob = other_pcsp_probs[other_pcsp]
            restricted_cond = restricted_scd[other_pcsp]
            restricted_cond_deriv = self.restricted_conditional_derivative(
                restriction=restriction, prob_of=other_pcsp,
                wrt=wrt, transit=transit,
                restricted_scd=restricted_scd
            )
            intermediate_result = -other_pcsp_prob * restricted_cond_deriv / restricted_cond
            result += intermediate_result
        return result

    def restricted_kl_gradient(self, other: 'SBN',
                               transit: dict = None,
                               restricted_self: 'SBN' = None,
                               restricted_subsplit_probs: dict = None,
                               other_subsplit_probs: dict = None,
                               res_transit: dict = None,
                               equiv_classes: dict = None):
        restriction = other.root_clade()
        root_subsplit = self.root_subsplit()
        if not restriction.issubset(self.root_clade()):
            raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
        if transit is None:
            transit = self.transit_probabilities()
        if restricted_self is None:
            restricted_self = self.restrict(restriction)
        if restricted_subsplit_probs is None:
            restricted_subsplit_probs = restricted_self.subsplit_probabilities()
        if other_subsplit_probs is None:
            other_subsplit_probs = other.subsplit_probabilities()
        if res_transit is None:
            res_transit = self.restrict_transit(restriction, transit)
        if equiv_classes is None:
            equiv_classes = self.equiv_classes(restriction)

        ancestor_factors = dict()
        cond_factors = dict()
        result = dict()
        for res_parent in other.iter_subsplits(include_root=True):
            n_dist_factor = len(list(res_parent.nontrivial_children()))
            ancestor_factors[res_parent] = ancestor_factor = (
                other_subsplit_probs[res_parent] / restricted_subsplit_probs[res_parent]
            )
            for wrt in self.iter_pcsp():
                if wrt not in result:
                    result[wrt] = 0.0
                wrt_parent = wrt.parent
                wrt_parent_clade = wrt.parent_clade()
                wrt_child = wrt.child
                wrt_parent_prob = transit.get(wrt_parent, dict()).get(root_subsplit, 0.0)
                wrt_cond_prob = self.get(wrt)
                wrt_child_to_res_parent = res_transit.get(res_parent, dict()).get(wrt_child, 0.0)
                wrt_parent_clade_to_res_parent = res_transit.get(res_parent, dict()).get(wrt_parent_clade, 0.0)
                result[wrt] += (
                        wrt_parent_prob * wrt_cond_prob * n_dist_factor * ancestor_factor
                        * (wrt_child_to_res_parent - wrt_parent_clade_to_res_parent)
                )
            for res_child in chain.from_iterable(other.get(res_parent)):
                pcsp_res = PCSP(res_parent, res_child)
                cond_factors[pcsp_res] = cond_factor = other.get(pcsp_res) / restricted_self.get(pcsp_res)
                for wrt in self.iter_pcsp():
                    wrt_parent = wrt.parent
                    wrt_parent_clade = wrt.parent_clade()
                    wrt_child = wrt.child
                    wrt_parent_prob = transit.get(wrt_parent, dict()).get(root_subsplit, 0.0)
                    wrt_cond_prob = self.get(wrt)
                    summation_factor1 = 0.0
                    summation_factor2 = 0.0
                    for ancestor in equiv_classes[res_parent]:
                        summation_factor1 += (
                                (transit.get(ancestor, dict()).get(wrt_child, 0.0)
                                 - transit.get(ancestor, dict()).get(wrt_parent_clade, 0.0))
                                * res_transit.get(res_child, dict()).get(ancestor, 0.0)
                        )
                        summation_factor2 += (
                                transit.get(ancestor, dict()).get(root_subsplit, 0.0)
                                * transit.get(wrt_parent, dict()).get(ancestor, 0.0)
                                * (res_transit.get(res_child, dict()).get(wrt_child, 0.0)
                                   - res_transit.get(res_child, dict()).get(wrt_parent_clade, 0.0))
                        )
                    result[wrt] += (
                            -wrt_parent_prob * wrt_cond_prob * ancestor_factor * cond_factor * summation_factor1
                            - wrt_cond_prob * ancestor_factor * cond_factor * summation_factor2
                    )
        return result

    def restricted_kl_gradient_multi(self, others: List['SBN'], weights: List[float] = None,
                                     transit: dict = None, restricted_selves: List['SBN'] = None, verbose=False):
        if weights is None:
            weights = [1] * len(others)
        if restricted_selves is None:
            restricted_selves = []
            for other in others:
                restriction = other.root_clade()
                restricted_selves.append(self.restrict(restriction))
        if not (len(others) == len(weights) == len(restricted_selves)):
            raise ValueError("Argument 'others', 'weights', and 'restricted_selves' not equal length or None.")
        if verbose:
            print("Checking for transit probabilities...")
        if transit is None:
            if verbose:
                print("Transit probabilities missing, calculating...")
            transit = self.transit_probabilities()
        if verbose:
            print("Transit probabilities found...")
        result = dict()
        for other, weight, restricted_self in zip(others, weights, restricted_selves):
            if verbose:
                print("Starting to examine reference with tip set ", other.root_clade())
                print("Calculating gradient...")
            grad = self.restricted_kl_gradient(other=other, transit=transit, restricted_self=restricted_self)
            for wrt in grad:
                if verbose:
                    print("  Examining parameter ", wrt)
                result[wrt] = result.get(wrt, 0.0) + grad[wrt] * weight
        return result

    def penalize(self, penalty: float):
        result = 0.0
        for parent_clade in self.iter_parents():
            dist = self.get(parent_clade)
            params_np = np.array(list(dist.params.values()))
            contrib = 0.5*penalty*np.sum((params_np - np.mean(params_np))**2)
            result += contrib
        return result

    def penalty_gradient(self, penalty: float):
        means = dict()
        for parent_clade in self.iter_parents():
            dist = self.get(parent_clade)
            params_np = np.array(list(dist.params.values()))
            means[parent_clade] = np.mean(params_np)
        result = dict()
        for pcsp in self.iter_pcsp():
            parent_clade = pcsp.parent_clade()
            mean = means.get(parent_clade)
            param = self.get_log(pcsp)
            result[pcsp] = penalty*(param - mean)
        return result

    def gradient_descent_pen(self, references: List['SBN'], penalty=0.0, starting_gamma=1.0, true_reference: 'SBN' = None,
                             max_iteration=50, verbose=True, const=0.5, shrink_factor=0.5, grow_factor=1.1):
        weights = [1] * len(references)
        restrictions = [reference.root_clade() for reference in references]
        current = self.copy()
        current_rs = [current.restrict(restriction) for restriction in restrictions]

        current_kl = sum(weight * reference.kl_divergence(current_r) for (weight, reference, current_r) in
                         zip(weights, references, current_rs))
        current_penalty = current.penalize(penalty)
        current_loss = current_kl + current_penalty

        current_transit = current.transit_probabilities()

        current_kl_grad = current.restricted_kl_gradient_multi(others=references, weights=weights,
                                                               transit=current_transit, restricted_selves=current_rs)
        current_penalty_grad = current.penalty_gradient(penalty)
        current_grad = {pcsp: current_kl_grad[pcsp] + current_penalty_grad[pcsp] for pcsp in current_kl_grad}
        current_grad_l2 = l2norm2(current_grad)

        gamma = starting_gamma
        loss_list = [current_loss]
        if true_reference is not None:
            true_kl_list = [true_reference.kl_divergence(current)]

        iteration = 0
        if verbose:
            print(f"Iter {iteration}: KL={current_loss:8.4g}")

        while iteration < max_iteration:
            iteration += 1
            candidate = current.copy()
            candidate.add_many_log(current_grad, -gamma)
            candidate.normalize()
            candidate_rs = [candidate.restrict(restriction) for restriction in restrictions]
            candidate_kl = sum(weight * reference.kl_divergence(candidate_r) for (weight, reference, candidate_r) in
                               zip(weights, references, candidate_rs))
            candidate_penalty = candidate.penalize(penalty)
            candidate_loss = candidate_kl + candidate_penalty

            if verbose:
                print(f"Iter {iteration}: KL={candidate_loss:8.4g}")
            # Armijo-Goldstein condition:
            # see https://math.stackexchange.com/questions/373868/optimal-step-size-in-gradient-descent
            if candidate_loss <= current_loss - const * gamma * current_grad_l2:
                gamma = gamma * grow_factor
            else:
                gamma = gamma * shrink_factor
                if verbose:
                    print(f"Shrinking gamma to {gamma}")
            if candidate_loss <= current_loss:
                current = candidate
                current_rs = candidate_rs
                current_loss = candidate_loss
                current_transit = current.transit_probabilities()
                current_kl_grad = current.restricted_kl_gradient_multi(others=references, weights=weights,
                                                                       transit=current_transit,
                                                                       restricted_selves=current_rs)
                current_penalty_grad = current.penalty_gradient(penalty)
                current_grad = {pcsp: current_kl_grad[pcsp] + current_penalty_grad[pcsp] for pcsp in current_kl_grad}
                current_grad_l2 = l2norm2(current_grad)
                # if verbose:
                #     print("Move accepted")
            else:
                if verbose:
                    print("Move rejected!")
            loss_list.append(current_loss)
            if true_reference is not None:
                true_kl_list.append(true_reference.kl_divergence(current))
        if verbose:
            print(f"Final KL={current_loss:8.4g}")
        if true_reference is not None:
            return current, loss_list, true_kl_list
        return current, loss_list

    @staticmethod
    def random(taxon_set, concentration=1.0, cutoff=0.0):
        result = SBN()
        root_clade = Clade(taxon_set)
        root_subsplit = Subsplit(root_clade)
        root_subsplit_clade = SubsplitClade(root_subsplit, root_clade)
        parent_clade_stack = [root_subsplit_clade]
        visited_parent_clades = set()
        while parent_clade_stack:
            parent_clade = parent_clade_stack.pop()
            if parent_clade in visited_parent_clades:
                continue
            visited_parent_clades.add(parent_clade)
            possible_subsplits = list(Subsplit.compatible_subsplits(parent_clade.clade))
            dist = ProbabilityDistribution.random(possible_subsplits, concentration=concentration, cutoff=cutoff)
            result.data[parent_clade] = dist
            for subsplit in dist:
                for child in subsplit.nontrivial_children():
                    parent_clade_stack.append(SubsplitClade(subsplit, child))
        return result

    @staticmethod
    def random_with_sparsity(taxon_set, sparsity, concentration=1.0):
        result = SBN()
        root_clade = Clade(taxon_set)
        root_subsplit = Subsplit(root_clade)
        root_subsplit_clade = SubsplitClade(root_subsplit, root_clade)
        parent_clade_stack = [root_subsplit_clade]
        visited_parent_clades = set()
        while parent_clade_stack:
            parent_clade = parent_clade_stack.pop()
            if parent_clade in visited_parent_clades:
                continue
            visited_parent_clades.add(parent_clade)
            possible_subsplits = list(Subsplit.compatible_subsplits(parent_clade.clade))
            dist = ProbabilityDistribution.random_with_sparsity(
                possible_subsplits, sparsity=sparsity, concentration=concentration
            )
            result.data[parent_clade] = dist
            for subsplit in dist:
                for child in subsplit.nontrivial_children():
                    parent_clade_stack.append(SubsplitClade(subsplit, child))
        return result

    @staticmethod
    def random_from_support(support: PCSPSupport, concentration: float = 1.0):
        result = SBN()
        root_subsplit_clade = support.root_subsplit_clade()
        parent_clade_stack = [root_subsplit_clade]
        visited_parent_clades = set()
        while parent_clade_stack:
            parent_clade = parent_clade_stack.pop()
            if parent_clade in visited_parent_clades:
                continue
            visited_parent_clades.add(parent_clade)
            children = support[parent_clade]
            dist = ProbabilityDistribution.random(support=children, concentration=concentration)
            result.data[parent_clade] = dist
            for subsplit in dist:
                for child in subsplit.nontrivial_children():
                    parent_clade_stack.append(SubsplitClade(subsplit, child))
        return result

    @staticmethod
    def from_tree_distribution(tree_distribution):
        result = SBN()
        for tree, prob in tree_distribution.items():
            for pcsp in tree.traverse_pcsps():
                result.add(pcsp, prob)
        result.normalize()
        return result


def binomial_min_one(n, p):
    if n <= 1:
        return 1
    new_p = max(0, (n*p - 1)/(n-1))
    return 1 + np.random.binomial(n-1, new_p)


def l2norm2(arg):
    vals = np.array(list(arg.values()))
    return np.sum(vals**2)
