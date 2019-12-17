import collections
from collections import defaultdict
from collections import namedtuple
from itertools import chain
from itertools import product
from itertools import combinations
from operator import itemgetter
from typing import List
import pprint
import random
import math

import numpy as np

import ete3


TreeRecursion = namedtuple("TreeRecursion", ['subsplit_set', 'clade_stack'])
# TreeRecursion2 = namedtuple("TreeRecursion2", ['support', 'clade_stack', 'tree_so_far', 'prob_so_far'])

# PCSS = namedtuple("PCSS", ['parent', 'child'])


class MyTree:
    def __init__(self, tree=None):
        if tree is None:
            self.tree = ete3.Tree()
        elif isinstance(tree, ete3.Tree):
            self.tree = tree.copy()
        elif isinstance(tree, str):
            self.tree = ete3.Tree(tree)
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

    def traverse(self, strategy='levelorder'):
        return self.tree.traverse(strategy=strategy)

    @staticmethod
    def _node_to_subsplit(node):
        assert len(node.children) == 2
        left_child, right_child = node.children
        return Subsplit(left_child.get_leaf_names(), right_child.get_leaf_names())

    def traverse_subsplits(self, strategy='levelorder'):
        for node in self.tree.traverse(strategy):
            if not node.is_leaf():
                yield self._node_to_subsplit(node)

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

    # Unfinished
    def label_restriction(self, restriction):
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


class Subsplit:
    def __init__(self, clade1, clade2=None):
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
        return (Subsplit(self.clade1 | other.clade1, self.clade2 | other.clade2),
                Subsplit(self.clade1 | other.clade2, self.clade2 | other.clade1))

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

    def valid_child(self, other):
        assert isinstance(other, Subsplit)
        return other.clade() in set(self.children())

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
    def parent_from_node(node):
        assert isinstance(node, ete3.TreeNode)
        if node.is_root():
            return Subsplit(Clade(node.get_leaf_names()))
        return Subsplit.from_node(node.up)


class PCSS:
    def __init__(self, parent, child):
        assert isinstance(parent, Subsplit) or isinstance(parent, Clade)
        assert isinstance(child, Subsplit)
        if isinstance(parent, Clade):
            parent = Subsplit(parent)
        self.parent = parent
        self.child = child

    def __repr__(self):
        return f"PCSS({repr(self.parent)}, {repr(self.child)})"

    def __str__(self):
        return f"{str(self.parent)}, {str(self.child)}"

    def __eq__(self, other):
        if not isinstance(other, PCSS):
            return False
        return self.parent == other.parent and self.child == other.child

    def __hash__(self):
        return hash((self.parent, self.child))

    def check(self):
        return self.parent.is_valid() and self.child.is_valid() and self.parent.valid_child(self.child)

    def is_valid(self):
        return self.check()

    def restrict(self, taxon_set):
        taxon_set = set(taxon_set)
        restricted_parent = self.parent.restrict(taxon_set)
        restricted_child = self.child.restrict(taxon_set)
        return PCSS(restricted_parent, restricted_child)

    @staticmethod
    def from_node(node):
        assert isinstance(node, ete3.TreeNode)
        return PCSS(Subsplit.parent_from_node(node), Subsplit.from_node(node))


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

    def add(self, subsplit):
        if not isinstance(subsplit, Subsplit):
            print("Error: Argument subsplit not class Subsplit")
            return
        if not subsplit.is_valid():
            print(f"Warning: subsplit is not valid, skipping")
            return
        if self.include_trivial or not subsplit.is_trivial():
            self.data[subsplit.clade()].add(subsplit)
            if self.include_trivial:
                self.data[subsplit.clade()].add(Subsplit(subsplit.clade()))

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
                if not subsplit.is_valid():
                    print(f"{subsplit} is not valid, skipping")
                    continue
                if self.include_trivial or not subsplit.is_trivial():
                    self.data[subsplit.clade()].add(subsplit)
                    if self.include_trivial:
                        self.data[subsplit.clade()].add(Subsplit(subsplit.clade()))
        except TypeError:
            print(f"Error: {arg} is not Subsplit or Iterable.")

    def clades(self):
        return self.data.keys()

    def get_taxon_set(self):
        return set().union(*self.data)

    def root_clade(self):
        return Clade(self.get_taxon_set())

    def is_complete(self):
        root_clade = self.root_clade()
        if root_clade not in self.data:
            return False
        clade_stack = [root_clade]
        while clade_stack:
            clade = clade_stack.pop()
            subsplits = self.data[clade]
            if not subsplits:
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
        result = SubsplitSupport(include_trivial=self.include_trivial)
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
        if not self.is_complete():
            print("Error: SubsplitSupport not complete, no tree possible.")
            return
        root_clade = Clade(self.get_taxon_set())
        tree = MyTree()
        clade_stack = [(tree.tree, root_clade)]  # MyTree alteration
        while clade_stack:
            node, clade = clade_stack.pop()
            possible_subsplits = [subsplit for subsplit in self.data[clade] if not subsplit.is_trivial()]
            assert len(possible_subsplits) > 0
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
        if not self.is_complete():
            print("Error: SubsplitSupport not complete, no tree possible.")
            return
        root_clade = Clade(self.get_taxon_set())
        support_list = []
        big_stack = [TreeRecursion(SubsplitSupport(), [root_clade])]
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
                    big_stack.append(TreeRecursion(new_support, new_clade_stack))
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
        assert isinstance(other, SubsplitSupport)
        result = SubsplitSupport(include_trivial=self.include_trivial)
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


class PCMiniSupport:
    def __init__(self, parent, items=None):
        assert isinstance(parent, Subsplit)
        self.parent = parent
        self.left = set()
        self.right = set()
        self.update(items)

    @property
    def data(self):
        return self.left | self.right

    def __repr__(self):
        pretty_repr = pprint.pformat(self.data)
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join((f"PCMiniSupport({repr(self.parent)}, ", pretty_repr, ")"))

    def __str__(self):
        return str(self.parent) + ": " + ", ".join(str(child) for child in self.data)

    def __iter__(self):
        yield from iter(self.data)

    def __getitem__(self, item):
        if not item:
            return set()
        elif item == self.parent.clade1:
            return self.left
        elif item == self.parent.clade2:
            return self.right
        else:
            raise KeyError

    def add(self, subsplit):
        assert isinstance(subsplit, Subsplit)
        clade = subsplit.clade()
        if clade == self.parent.clade1:
            self.left.add(subsplit)
        elif clade == self.parent.clade2:
            self.right.add(subsplit)
        else:
            print(f"Error: subsplit ({str(subsplit)}) is not compatible with either side of parent ({str(self.parent)}). Skipping.")

    def remove(self, subsplit):
        assert isinstance(subsplit, Subsplit)
        clade = subsplit.clade()
        if clade == self.parent.clade1:
            self.left.remove(subsplit)
        elif clade == self.parent.clade2:
            self.right.remove(subsplit)
        else:
            print(f"Error: subsplit ({str(subsplit)}) is not compatible with either side of parent ({str(self.parent)}). Skipping.")

    def update(self, items):
        if items is None:
            return
        for item in items:
            self.add(items)

    def locally_complete(self):
        return (self.left or len(self.parent.clade1) <= 1) and (self.right or len(self.parent.clade2) <= 1)

    def to_set(self):
        return self.to_pcss_set()

    def to_pcss_set(self):
        return {PCSS(self.parent, child) for child in self}


# TODO: Lots of work here
class PCSSSupport:
    def __init__(self, arg=None, include_trivial=False):
        self.data = dict()
        self.include_trivial = include_trivial
        self.update(arg)

    def __repr__(self):
        pretty_repr = pprint.pformat(self.to_set())
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join(("PCSSSupport(", pretty_repr, ")"))

    def __str__(self):
        return str(self.to_string_dict())
        # return pprint.pformat(set(map(str, self.to_set())))

    def __iter__(self):
        yield from iter(self.to_set())

    def __getitem__(self, parent):
        assert isinstance(parent, Subsplit)
        if parent not in self.data:
            return PCMiniSupport(parent)
        return self.data[parent]

    def __eq__(self, other):
        if not isinstance(other, PCSSSupport):
            return False
        return self.to_set() == other.to_set()

    def add(self, pcss):
        if not isinstance(pcss, PCSS):
            print("Error: Argument subsplit not class PCSS")
            return
        if not pcss.is_valid():
            print(f"Warning: argument pcss ({str(pcss)}) is not valid, skipping")
            return
        if pcss.parent not in self.data:
            self.data[pcss.parent] = PCMiniSupport(pcss.parent)
        self.data[pcss.parent].add(pcss.child)

    def update(self, arg):
        if arg is None:
            return
        if isinstance(arg, PCSS):
            arg = [arg]
        if isinstance(arg, PCSSSupport):
            arg = arg.to_set()
        try:
            for pcss in arg:
                if not isinstance(pcss, PCSS):
                    print(f"{pcss} is not a PCSS, skipping")
                    continue
                self.add(pcss)
        except TypeError:
            print(f"Error: {arg} is not PCSS or Iterable.")

    def get_taxon_set(self):
        result = set()
        for parent in self.data:
            result |= parent.clade()
        return result

    def root_clade(self):
        return Clade(self.get_taxon_set())

    def is_complete(self, verbose=False):
        result = True
        root_clade = Subsplit(self.root_clade())
        if root_clade not in self.data:
            if verbose:
                print("No root splits found.")
            result = False
        parent_stack = [root_clade]
        while parent_stack:
            parent = parent_stack.pop()
            mini_support = self[parent]
            if not mini_support.locally_complete():
                if verbose:
                    print(f"Not locally complete: {mini_support}")
                result = False
            for subsplit in mini_support:
                parent_stack.append(subsplit)
        return result

    def valid_parent(self, subsplit):
        if not isinstance(subsplit, Subsplit):
            return False
        return subsplit in self.data.keys()

    def restrict(self, taxon_set):
        result = PCSSSupport(include_trivial=self.include_trivial)
        for parent in self.data:
            result.update(pcss.restrict(taxon_set) for pcss in self[parent].to_pcss_set())
        return result

    def to_set(self):
        result = set()
        for parent in self.data:
            result.update(self[parent].to_pcss_set())
        return result

    def to_string_dict(self):
        return {str(parent): set(map(str, self[parent])) for parent in self.data}

    def to_string_set(self):
        return set(map(str, self.to_set()))

    def random_tree(self):
        if not self.is_complete():
            print("Error: PCSSSupport not complete, no tree possible.")
            return
        root_clade = self.root_clade()
        tree = MyTree()
        parent_stack = [(tree.tree, Subsplit(root_clade), root_clade)]
        while parent_stack:
            node, parent, clade = parent_stack.pop()
            possible_subsplits = [subsplit for subsplit in self[parent][clade]
                                  if not subsplit.is_trivial()]
            assert possible_subsplits
            subsplit = random.choice(possible_subsplits)
            node.name = str(subsplit)
            left = subsplit.clade1
            right = subsplit.clade2
            assert len(left) > 0 and len(right) > 0
            if len(left) > 1:
                left_node = node.add_child()
                parent_stack.append((left_node, subsplit, left))
            else:
                node.add_child(name=str(left))
            if len(right) > 1:
                right_node = node.add_child()
                parent_stack.append((right_node, subsplit, right))
            else:
                node.add_child(name=str(right))
        return tree

    def all_strictly_complete_supports(self, verbose=False):
        # if not self.is_complete():
        #     print("Error: PCSSSupport not complete, no tree possible.")
        #     return
        root_clade = self.root_clade()
        support_list = []
        big_stack = [(PCSSSupport(), [(Subsplit(root_clade), root_clade)])]
        while big_stack:
            current_support, parent_stack = big_stack.pop()
            if verbose:
                print(f"Current subsplit set: {current_support}")
                print(f"Current parent stack: {parent_stack}")
            possibilities = []
            for parent, clade in parent_stack:
                if verbose:
                    print(f"Examining parent: {parent}")
                    print(f"Examining clade: {clade}")
                possibilities.append([PCSS(parent, subsplit) for subsplit in self[parent][clade] if not subsplit.is_trivial()])
            if verbose:
                print(f"Current possibilities: {possibilities}")
            for new_pcsss in product(*possibilities):
                if verbose:
                    print(f"Examining possibility: {new_pcsss}")
                new_support = PCSSSupport(current_support)
                new_parent_stack = []
                for pcss in new_pcsss:
                    new_support.add(pcss)
                    for child_clade in pcss.child.nontrivial_children():
                        new_parent_stack.append((pcss.child, child_clade))
                        if verbose:
                            print(f"Pushing parent: {pcss.child}")
                            print(f"Pushing clade: {child_clade}")
                if new_parent_stack:
                    big_stack.append((new_support, new_parent_stack))
                    if verbose:
                        print(f"Pushing subsplit set: {new_support}")
                else:
                    if new_support.is_complete():
                        if verbose:
                            print(f"Sending to output: {new_support}")
                        support_list.append(new_support)
                    else:
                        if verbose:
                            print(f"Incomplete support, NOT sending to output: {new_support}")
        return support_list

    def all_trees(self):
        strict_supports = self.all_strictly_complete_supports()
        return [support.random_tree() for support in strict_supports]

    def mutualize(self, other):
        assert isinstance(other, PCSSSupport)
        result = PCSSSupport(include_trivial=self.include_trivial)
        root_clade1 = self.root_clade()
        root_clade2 = other.root_clade()
        big_clade = root_clade1 | root_clade2
        parent_stack = [(Subsplit(big_clade), Subsplit(root_clade1), Subsplit(root_clade2), big_clade)]
        while parent_stack:
            parent, restricted_parent1, restricted_parent2, clade = parent_stack.pop()
            restricted_clade1 = clade & root_clade1
            restricted_clade2 = clade & root_clade2
            candidate_subsplits1 = self[restricted_parent1][restricted_clade1] | {Subsplit(restricted_clade1)}
            candidate_subsplits2 = other[restricted_parent2][restricted_clade2] | {Subsplit(restricted_clade2)}
            candidate_big_subsplits = chain.from_iterable(
                (big_subsplit for big_subsplit in subsplit1.cross(subsplit2) if big_subsplit.is_strictly_valid())
                for subsplit1, subsplit2 in product(candidate_subsplits1, candidate_subsplits2)
            )
            for big_subsplit in candidate_big_subsplits:
                result.add(PCSS(parent, big_subsplit))
                new_restricted_parent1 = big_subsplit.restrict(root_clade1)
                if new_restricted_parent1.is_trivial() and not self.valid_parent(new_restricted_parent1):
                    new_restricted_parent1 = restricted_parent1
                new_restricted_parent2 = big_subsplit.restrict(root_clade2)
                if new_restricted_parent2.is_trivial() and not other.valid_parent(new_restricted_parent2):
                    new_restricted_parent2 = restricted_parent2
                for child_clade in big_subsplit.nontrivial_children():
                    parent_stack.append((big_subsplit, new_restricted_parent1, new_restricted_parent2, child_clade))
        return result

    def add_tree(self, tree):
        if isinstance(tree, MyTree):
            root_clade = tree.root_clade()
        elif isinstance(tree, ete3.Tree):
            root_clade = Clade(tree.get_leaf_names())
        else:
            print("Error: tree not ete3.Tree or MyTree.")
            return
        for node in tree.traverse():
            pcss = PCSS.from_node(node)
            if not pcss.child.is_trivial():
                self.add(pcss)

    @staticmethod
    def from_tree(tree):
        # if isinstance(tree, MyTree):
        #     tree = tree.tree
        result = PCSSSupport()
        result.add_tree(tree)
        return result

    @staticmethod
    def from_trees(trees):
        # if isinstance(tree, MyTree):
        #     tree = tree.tree
        result = PCSSSupport()
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
        return self.normalized

    def copy(self):
        return self.__copy__()

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
        self._require_normalization()
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

    def support(self):
        return set(self.params.keys())

    def update_lin(self, arg):
        if arg is None:
            return
        if not isinstance(arg, collections.Mapping):
            print("Argument arg not a Mapping.")
            return
        for item in arg:
            self.set_lin(item, arg[item])

    def update_log(self, arg):
        if arg is None:
            return
        if not isinstance(arg, collections.Mapping):
            print("Argument arg not a Mapping.")
            return
        for item in arg:
            self.set_log(item, arg[item])

    def update(self, arg=None, log=False):
        if arg is None:
            return
        if log:
            self.update_log(arg)
        else:
            self.update_lin(arg)

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

    # TODO: max_item, max_(log)_likelihood, kl_divergence, random

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


class ProbabilityDistributionOld(collections.Mapping):
    def __init__(self, arg=None):
        self.data = dict()
        self.update(arg)

    def __contains__(self, item):
        return self.data.__contains__(item)

    def __eq__(self, other):
        return self.items() == other.items()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item):
        return self.data.get(item, 0.0)

    def __iter__(self):
        return iter(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __setitem__(self, item, probability):
        self.set(item, probability)

    def __repr__(self):
        return "ProbabilityDistribution("+repr(self.data)+")"

    def __str__(self):
        strs = [str(key) + ": " + str(round(self[key], 4)) for key in self]
        return "{" + ", ".join(strs) + "}"

    def add(self, item, probability):
        if not (0 <= probability <= 1):
            print(f"Error: probability ({probability}) out of bounds [0, 1]")
            return
        if item not in self:
            self[item] = 0.0
        self[item] += probability

    def check(self):
        return (all(0 <= prob <= 1 for prob in self.data.values())
                and sum(self.data.values()) == 1.0)

    def get(self, item):
        return self.data.get(item, 0.0)

    def remove(self, item):
        self.data.pop(item, None)

    def sample(self):
        items, probabilities = zip(*self.data.items())
        return random.choices(population=items, weights=probabilities)[0]

    def samples(self, k=1):
        items, probabilities = zip(*self.data.items())
        return random.choices(population=items, weights=probabilities, k=k)

    def set(self, item, probability):
        if not (0 <= probability <= 1):
            print(f"Error: probability ({probability}) out of bounds [0, 1]")
            return
        self.data[item] = probability

    def support(self):
        return {item for item in self.data if self.data[item] > 0.0}

    def update(self, arg=None):
        if arg is None:
            return
        if not isinstance(arg, collections.Mapping):
            print("Argument arg not a Mapping.")
            return
        for item in arg:
            self.set(item, arg[item])

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def normalize(self):
        norm = sum(self.values())
        return ProbabilityDistribution({item: prob / norm for item, prob in self.items()})

    def trim_small(self, cutoff):
        return ProbabilityDistribution({item: prob for item, prob in self.items() if prob > cutoff})

    def likelihood(self, item):
        return self.get(item)

    def log_likelihood(self, item):
        lik = self.likelihood(item)
        if lik > 0.0:
            return math.log(lik)
        else:
            return -math.inf

    def max_item(self):
        return max(self.items(), key=itemgetter(1))

    def max_likelihood(self):
        return self.max_item()[0]

    def kl_divergence(self, other):
        result = 0.0
        for item in self:
            log_p = self.log_likelihood(item)
            log_q = other.log_likelihood(item)
            p = math.exp(log_p)
            if p > 0.0:
                result += -p*(log_q - log_p)
        return result

    @staticmethod
    def random(support, concentration=1.0, cutoff=0.0):
        distribution = np.random.dirichlet([concentration] * len(support))
        pr = ProbabilityDistribution({item: prob for item, prob in zip(support, distribution)})
        pr = pr.trim_small(cutoff)
        pr = pr.normalize()
        return pr


class PCMiniSet:
    def __init__(self, parent, arg=None):
        assert isinstance(parent, Subsplit)
        self.parent = parent
        self.left = ProbabilityDistribution()
        self.right = ProbabilityDistribution()
        self.update(arg)

    # @property
    # def data(self):
    #     return self.left | self.right

    def __repr__(self):
        pretty_repr = pprint.pformat(self.data)
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join((f"PCMiniSet({repr(self.parent)}, ", pretty_repr, ")"))

    def __str__(self):
        return str(self.parent) + ": " + ", ".join(str(child) for child in self.data)

    def __iter__(self):
        yield from iter(self.data)

    def __getitem__(self, item):
        if not item:
            return set()
        elif item == self.parent.clade1:
            return self.left
        elif item == self.parent.clade2:
            return self.right
        else:
            raise KeyError

    def add(self, subsplit, probability):
        assert isinstance(subsplit, Subsplit)
        clade = subsplit.clade()
        if clade == self.parent.clade1:
            self.left.add(subsplit)
        elif clade == self.parent.clade2:
            self.right.add(subsplit)
        else:
            print(f"Error: subsplit ({str(subsplit)}) is not compatible with either side of parent ({str(self.parent)}). Skipping.")

    def remove(self, subsplit):
        assert isinstance(subsplit, Subsplit)
        clade = subsplit.clade()
        if clade == self.parent.clade1:
            self.left.remove(subsplit)
        elif clade == self.parent.clade2:
            self.right.remove(subsplit)
        else:
            print(f"Error: subsplit ({str(subsplit)}) is not compatible with either side of parent ({str(self.parent)}). Skipping.")

    def update(self, arg):
        if arg is None:
            return
        for item in arg:
            self.add(item, arg[item])

    def locally_complete(self):
        return (self.left or len(self.parent.clade1) <= 1) and (self.right or len(self.parent.clade2) <= 1)

    def support(self):
        return self.left.support() | self.right.support()

    def sample(self):
        return self.left.sample(), self.right.sample()


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

    def to_ccd(self):
        result = CCDSet()
        for tree, prob in self.items():
            support = tree.to_support()
            for subsplit in support:
                result.add(subsplit, prob)
        result.normalize()
        return result

    # TODO: Consider merging these two functions
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


class CCDSet:
    def __init__(self, cond_probs=None):
        self.data = dict()
        self.update(cond_probs)

    def __copy__(self):
        result = CCDSet()
        for clade in self.clades():
            result.data[clade] = self.data[clade].copy()
        return result

    def __eq__(self, other):
        if not isinstance(other, CCDSet):
            return False
        if not self.clades() == other.clades():
            return False
        for clade in self.clades():
            if not self[clade] == other[clade]:
                return False
        return True

    def __repr__(self):
        # return "CCDSet(" + repr(dict(self.data)) + ")"
        pretty_repr = pprint.pformat(dict(self.data))
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join(("CCDSet(", pretty_repr, ")"))

    def __str__(self):
        strs = [str(clade) + ": " + str(self.get(clade)) for clade in sorted(self.clades())]
        return "\n".join(strs)

    def __getitem__(self, item):
        if isinstance(item, Clade):
            return self.data.get(item, ProbabilityDistribution())
        if isinstance(item, Subsplit):
            clade = item.clade()
            return self.data.get(clade, ProbabilityDistribution()).get(item)

    def __len__(self):
        result = 0
        for clade in self.clades():
            result += len(self[clade])
        return result

    # TODO: Consider adding __setitem__

    # def add(self, subsplit, probability):
    #     if not isinstance(subsplit, Subsplit):
    #         raise TypeError("Argument 'subsplit' not class Subsplit")
    #     clade = subsplit.clade()
    #     if clade not in self.data:
    #         self.data[clade] = ProbabilityDistribution()
    #     self.data[clade][subsplit] += probability

    def add(self, subsplit, value, log=False):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not class Subsplit")
        if log:
            self.add_log(subsplit, value)
        else:
            self.add_lin(subsplit, value)

    def add_lin(self, subsplit, prob):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not class Subsplit")
        if prob == 0.0:
            return
        clade = subsplit.clade()
        if clade not in self.data:
            self.data[clade] = ProbabilityDistribution()
        self.data[clade].add_lin(subsplit, prob)

    def add_log(self, subsplit, log_prob):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not class Subsplit")
        if log_prob == 0.0:
            return
        clade = subsplit.clade()
        if clade not in self.data:
            # self.data[clade] = ProbabilityDistribution()
            return
        self.data[clade].add_log(subsplit, log_prob)

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
        for clade in self.data:
            if not self.data[clade].check():
                result.add(clade)
        return result

    def check(self):
        return self.check_distributions() and self.support().is_complete()

    def check_distributions(self):
        for clade in self.data:
            if not self.data[clade].check():
                return False
        return True

    def clades(self):
        return self.data.keys()

    def clade_probabilities(self):
        clades = sorted(self.clades(), key=lambda W: (-len(W), W))
        result = dict()
        result[self.root_clade()] = 1.0
        for clade in clades:
            for subsplit in self[clade]:
                uncond_prob = result[clade] * self[subsplit]
                for child_clade in subsplit.nontrivial_children():
                    if child_clade not in result:
                        result[child_clade] = 0.0
                    result[child_clade] += uncond_prob
        return result

    def clade_to_clade_probabilities(self):
        support = self.support()
        clades = sorted(support.clades(), key=lambda clade: (-len(clade), clade))
        result = {self.root_clade(): dict()}
        for clade in clades:
            if clade not in result:
                result[clade] = dict()
            result[clade][clade] = 1.0
            curr_paths = result[clade]
            for subsplit in support[clade]:
                cond_prob = self[subsplit]
                for parent_clade in curr_paths.keys():
                    path_prob = curr_paths[parent_clade] * cond_prob
                    for child_clade in subsplit.nontrivial_children():
                        if child_clade not in result:
                            result[child_clade] = dict()
                        if parent_clade not in result[child_clade]:
                            result[child_clade][parent_clade] = 0.0
                        result[child_clade][parent_clade] += path_prob
        return result

    def copy(self):
        return self.__copy__()

    def get(self, item):
        if isinstance(item, Clade):
            return self.data.get(item, ProbabilityDistribution())
        if isinstance(item, Subsplit):
            clade = item.clade()
            return self.data.get(clade, ProbabilityDistribution()).get(item)

    # TODO: Bug here since self.get can return a ProbabilityDistribution
    def get_log(self, item):
        prob = self.get(item)

        if prob > 0.0:
            return math.log(prob)
        else:
            return -math.inf

    def is_complete(self):
        return self.support().is_complete()

    def iter_clades(self):
        yield from self.clades()

    def iter_probabilities(self):
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, clade_prob = clade_stack.pop()
            cond_dist = self[clade]
            for subsplit in cond_dist:
                cond_prob = self[subsplit]
                joint_prob = clade_prob * cond_prob
                yield (subsplit, cond_prob, joint_prob)
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, joint_prob))

    def iter_subsplits(self):
        for clade in self.iter_clades():
            yield from self.data[clade]

    def iter_unconditional_probabilities(self):
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, clade_prob = clade_stack.pop()
            cond_dist = self[clade]
            for subsplit in cond_dist:
                cond_prob = self[subsplit]
                joint_prob = clade_prob * cond_prob
                yield (subsplit, joint_prob)
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, joint_prob))

    def kl_divergence(self, other):
        if isinstance(other, CCDSet):
            return self.kl_divergence_ccd(other)
        if isinstance(other, TreeDistribution):
            return self.kl_divergence_treedist(other)
        print("Error: argument 'other' not a CCDNet or TreeDistribution")

    def kl_divergence_ccd(self, other):
        result = 0.0
        for subsplit, cond_prob, joint_prob in self.iter_probabilities():
            result += -joint_prob * (other.get_log(subsplit) - math.log(cond_prob))
        return result

    def kl_divergence_treedist(self, other):
        result = 0.0
        for tree in other:
            log_q = other.log_likelihood(tree)
            log_p = self.log_likelihood(tree)
            result += -math.exp(log_p) * (log_q - log_p)
        return result

    def likelihood(self, tree, strategy='levelorder'):
        return math.exp(self.log_likelihood(tree, strategy))

    def log_likelihood(self, tree, strategy='levelorder'):
        result = 0.0
        for subsplit in tree.traverse_subsplits(strategy):
            cond_prob = self[subsplit]
            if cond_prob > 0.0:
                result += math.log(cond_prob)
            else:
                result += -math.inf
        return result

    def max_clade_tree(self):
        clade_probs = self.clade_probabilities()
        clades = sorted(self.clades(), key=lambda W: (len(W), W))
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

    def max_prob_tree(self):
        clades = sorted(self.clades(), key=lambda W: (len(W), W))
        best_subtree_and_prob = {}
        for clade in clades:
            best_subsplit = None
            best_subtree_prob = 0.0
            for subsplit in self[clade]:
                cond_prob = self[subsplit]
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

    def normalize(self, clade=None):
        if clade is None:
            for clade in self.data:
                self.data[clade].normalize()
        elif isinstance(clade, Clade):
            self.data[clade].normalize()
        else:
            raise TypeError("Argument 'clade' not Clade or None")

    def random_tree(self):
        if not self.is_complete():
            print("Error: CCDSet not complete, no tree possible.")
            return
        root_clade = self.root_clade()
        tree = MyTree()
        clade_stack = [(tree.tree, root_clade)]  # MyTree alteration
        while clade_stack:
            node, clade = clade_stack.pop()
            # possible_subsplits = [subsplit for subsplit in self.data[clade] if not subsplit.is_trivial()]
            # assert len(possible_subsplits) > 0
            assert self.data[clade].check()
            subsplit = self.data[clade].sample()
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

    def remove(self, subsplit):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument subsplit not class Subsplit")
        clade = subsplit.clade()
        self.data[clade].remove(subsplit)

    def restrict(self, taxon_set: object, verbose: object = False) -> object:
        taxon_set = Clade(taxon_set)
        result = CCDSet()
        clade_probabilities = self.clade_probabilities()
        # clade_stack = [self.root_clade()]
        # visited_clades = set()
        for clade, prob in clade_probabilities.items():
            for subsplit in self[clade]:
                subsplit_prob = self[subsplit]
                restricted_subsplit = subsplit.restrict(taxon_set)
                if not restricted_subsplit.is_trivial():
                    result.add(restricted_subsplit, clade_probabilities[clade] * subsplit_prob)
        result.normalize()
        return result

    def root_clade(self):
        return Clade(self.taxon_set())

    # def set(self, subsplit, probability):
    #     if not isinstance(subsplit, Subsplit):
    #         raise TypeError("Argument subsplit not class Subsplit")
    #     clade = subsplit.clade()
    #     if clade not in self.data:
    #         self.data[clade] = ProbabilityDistribution()
    #     self.data[clade][subsplit] = probability

    def set(self, subsplit, value, log=False):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not class Subsplit")
        if log:
            self.set_log(subsplit, value)
        else:
            self.set_lin(subsplit, value)

    def set_lin(self, subsplit, prob):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not class Subsplit")
        clade = subsplit.clade()
        if prob > 0.0 and clade not in self.data:
            self.data[clade] = ProbabilityDistribution()
        self.data[clade].set_lin(subsplit, prob)

    def set_log(self, subsplit, log_prob):
        if not isinstance(subsplit, Subsplit):
            raise TypeError("Argument 'subsplit' not class Subsplit")
        clade = subsplit.clade()
        if log_prob > -math.inf and clade not in self.data:
            self.data[clade] = ProbabilityDistribution()
        self.data[clade].set_log(subsplit, log_prob)

    def set_many(self, arg, log=False):
        if log:
            self.set_many_log(arg)
        else:
            self.set_many_lin(arg)

    def set_many_lin(self, arg):
        for key in arg:
            self.set_lin(key, arg[key])

    def set_many_log(self, arg):
        for key in arg:
            self.set_log(key, arg[key])

    def support(self, include_trivial=False):
        result = SubsplitSupport(include_trivial=include_trivial)
        for clade in self.data:
            # result.add({subsplit for subsplit in self.data[clade] if self.data[clade][subsplit] > 0})
            result.update(self.data[clade].support())
        return result

    def taxon_set(self):
        return set().union(*self.data)

    def tree_distribution(self, verbose=False):
        if not self.is_complete():
            print("Error: CCDSet not complete, no tree possible.")
            return
        root_clade = self.root_clade()
        tree = MyTree()
        tree.tree.clade = root_clade
        result = TreeDistribution()
        big_stack = [(tree.tree, 0.0)]
        while big_stack:
            current_tree, log_prob_so_far = big_stack.pop()
            if verbose:
                print(f"Current tree: {current_tree}")
                print(f"Current log-probability: {log_prob_so_far}")
            possibilities = []
            clade_stack = [leaf.clade for leaf in current_tree.iter_leaves() if len(leaf.clade) > 1]
            if not clade_stack:
                result.set(MyTree(current_tree), math.exp(log_prob_so_far))
                if verbose:
                    print(f"Sending to output: {current_tree} with probability {math.exp(log_prob_so_far)}")
                continue
            for clade in clade_stack:
                if verbose:
                    print(f"Examining clade: {clade}")
                possibilities.append(subsplit for subsplit, prob in self.data[clade].items()
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
                    new_log_prob += math.log(self.get(subsplit))
                    for child_clade in subsplit.children():
                        child_node = node.add_child(name=str(child_clade))
                        child_node.clade = child_clade
                        if verbose:
                            print(f"Pushing clade: {child_clade}")
                big_stack.append((new_tree, new_log_prob))
                if verbose:
                    print(f"Pushing tree: {new_tree} with probability {math.exp(new_log_prob)}")
        result.normalize()
        return result

    def unconditional_probabilities(self):
        result = defaultdict(float)
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, clade_prob = clade_stack.pop()
            cond_dist = self[clade]
            for subsplit in cond_dist:
                cond_prob = self[subsplit]
                joint_prob = clade_prob * cond_prob
                result[subsplit] += joint_prob
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, joint_prob))
        return result

    def uncond_prob_derivative(self, prob_of: Subsplit, wrt: Subsplit, c2c: dict = None):
        root_clade = self.root_clade()
        clade1 = wrt.clade()
        clade2 = prob_of.clade()
        if not clade2.issubset(clade1):
            return 0.0
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        uncond_wrt = c2c.get(clade1, dict()).get(root_clade, 0.0) * self[wrt]
        if clade1 == clade2:
            indic = 1.0 if (prob_of == wrt) else 0.0
            return uncond_wrt * (indic - self[prob_of])
        comp_child = wrt.compatible_child(prob_of)
        comp_child_to_clade2 = 0.0
        if comp_child is not None:
            comp_child_to_clade2 = c2c.get(clade2, dict()).get(comp_child, 0.0)
        clade1_to_clade2 = c2c.get(clade2, dict()).get(clade1, 0.0)
        return uncond_wrt * (comp_child_to_clade2 - clade1_to_clade2) * self[prob_of]

    def restricted_uncond_prob_derivative(self, restriction, prob_of: Subsplit, wrt: Subsplit, c2c: dict = None):
        restriction = Clade(restriction)
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        result = 0.0
        for subsplit in self.iter_subsplits():
            # print(f"Examining subsplit {subsplit}:")
            if subsplit.restrict(restriction) == prob_of:
                # print("Found a match")
                result += self.uncond_prob_derivative(prob_of=subsplit, wrt=wrt, c2c=c2c)
        return result

    def restricted_clade_prob_derivative(self, restriction, prob_of: Clade, wrt: Subsplit, c2c: dict = None):
        restriction = Clade(restriction)
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        result = 0.0
        for subsplit in self.iter_subsplits():
            # print(f"Examining subsplit {subsplit}:")
            if subsplit.restrict(restriction).clade() == prob_of and not subsplit.restrict(restriction).is_trivial():
                # print("Found a match")
                result += self.uncond_prob_derivative(prob_of=subsplit, wrt=wrt, c2c=c2c)
        return result

    def restricted_cond_prob_derivative(self, restriction, prob_of: Subsplit, wrt: Subsplit,
                                        c2c: dict = None, restricted_self: 'CCDSet' = None):
        restriction = Clade(restriction)
        clade = prob_of.clade()
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        if restricted_self is None:
            restricted_self = self.restrict(restriction)
        restricted_uncond_probs = restricted_self.unconditional_probabilities()
        restricted_clade_probs = restricted_self.clade_probabilities()
        # Quotient rule d(top/bot) = (bot*dtop-top*dbot) / bot**2
        top = restricted_uncond_probs[prob_of]
        bot = restricted_clade_probs[clade]
        dtop = self.restricted_uncond_prob_derivative(restriction, prob_of, wrt, c2c)
        dbot = self.restricted_clade_prob_derivative(restriction, clade, wrt, c2c)
        return (bot*dtop - top*dbot) / bot**2

    def restricted_kl_divergence_derivative_old(self, other: 'CCDSet', wrt: Subsplit,
                                                c2c: dict = None, restricted_self: 'CCDSet' = None, verbose_dict: dict = None):
        restriction = other.root_clade()
        if not restriction.issubset(self.root_clade()):
            raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        if restricted_self is None:
            restricted_self = self.restrict(restriction)
        verbose = False
        if isinstance(verbose_dict, dict):
            verbose = True
        other_uncond = other.unconditional_probabilities()
        result = 0.0
        for sbar in other.iter_subsplits():
            sbar_other_uncond = other_uncond[sbar]
            sbar_self_cond = restricted_self[sbar]
            sbar_self_cond_deriv = self.restricted_cond_prob_derivative(restriction, sbar, wrt, c2c, restricted_self)
            result += -sbar_other_uncond*sbar_self_cond_deriv/sbar_self_cond
            if verbose:
                verbose_dict[sbar] = {'P': sbar_other_uncond, 'q': sbar_self_cond, 'dq': sbar_self_cond_deriv}
        return result

    def restricted_kl_divergence_derivative(self, other: 'CCDSet', wrt: Subsplit,
                                            c2c: dict = None, restricted_self: 'CCDSet' = None, verbose_dict: dict = None):
        restriction = other.root_clade()
        if not restriction.issubset(self.root_clade()):
            raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        if restricted_self is None:
            restricted_self = self.restrict(restriction)
        restricted_uncond = restricted_self.unconditional_probabilities()
        restricted_clade = restricted_self.clade_probabilities()
        other_uncond = other.unconditional_probabilities()
        other_clade = other.clade_probabilities()
        verbose = False
        if isinstance(verbose_dict, dict):
            verbose = True
        result = 0.0
        for ss in self.iter_subsplits():
            ss_r = ss.restrict(restriction)
            if ss_r.is_trivial():
                continue
            U_r = ss_r.clade()
            ss_factor = other_uncond[ss_r]/restricted_uncond[ss_r] - other_clade[U_r]/restricted_clade[U_r]
            deriv = self.uncond_prob_derivative(ss, wrt, c2c)
            result += -deriv * ss_factor
            if verbose:
                verbose_dict[ss] = {'deriv': deriv, 'ss_factor': ss_factor}
        return result

    def restricted_kl_divergence_gradient(self, other: 'CCDSet', c2c: dict = None, restricted_self: 'CCDSet' = None):
        restriction = other.root_clade()
        if not restriction.issubset(self.root_clade()):
            raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        if restricted_self is None:
            restricted_self = self.restrict(restriction)
        restricted_uncond = restricted_self.unconditional_probabilities()
        restricted_clade = restricted_self.clade_probabilities()
        other_uncond = other.unconditional_probabilities()
        other_clade = other.clade_probabilities()
        # verbose = False
        # if isinstance(verbose_dict, dict):
        #     verbose = True
        result = dict()
        for ss in self.iter_subsplits():
            U = ss.clade()
            if U not in c2c:
                continue
            ss_r = ss.restrict(restriction)
            if ss_r.is_trivial():
                continue
            U_r = ss_r.clade()
            ss_factor = other_uncond[ss_r]/restricted_uncond[ss_r] - other_clade[U_r]/restricted_clade[U_r]
            for U_prime in c2c[U]:
                # TODO: Consider converting .get() to differentiate
                #  getting a conditional probability from getting a
                #  ProbabilityDistribution
                for wrt in self.get(U_prime):
                    deriv = self.uncond_prob_derivative(ss, wrt, c2c)
                    if wrt not in result:
                        result[wrt] = 0.0
                    result[wrt] += -deriv * ss_factor
        return result

    def restricted_kl_divergence_gradient_multi(self, others: List['CCDSet'], weights: List[float] = None,
                                                c2c: dict = None, restricted_selves: List['CCDSet'] = None):
        if weights is None:
            weights = [1] * len(others)
        if restricted_selves is None:
            restricted_selves = []
            for other in others:
                restriction = other.root_clade()
                restricted_selves.append(self.restrict(restriction))
        assert len(others) == len(weights) == len(restricted_selves)
        if c2c is None:
            c2c = self.clade_to_clade_probabilities()
        result = dict()
        for other, weight, restricted_self in zip(others, weights, restricted_selves):
            grad = self.restricted_kl_divergence_gradient(other=other, c2c=c2c, restricted_self=restricted_self)
            for wrt in grad:
                if wrt not in result:
                    result[wrt] = 0.0
                result[wrt] += grad[wrt] * weight
        return result

    def update(self, cond_probs):
        if cond_probs is None:
            return
        if not isinstance(cond_probs, collections.Mapping):
            raise TypeError("Argument cond_probs not a Mapping")
        for subsplit in cond_probs:
            if not isinstance(subsplit, Subsplit):
                print("Warning: Non-Subsplit found in cond_probs, skipping")
                continue
            clade = subsplit.clade()
            if clade not in self.data:
                self.data[clade] = ProbabilityDistribution()
            self.data[clade][subsplit] = cond_probs[subsplit]

    @staticmethod
    def random(taxon_set, concentration=1.0, cutoff=0.0):
        result = CCDSet()
        root_clade = Clade(taxon_set)
        clade_stack = [root_clade]
        visited_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in visited_clades:
                continue
            visited_clades.add(clade)
            possible_subsplits = list(Subsplit.compatible_subsplits(clade))
            dist = ProbabilityDistribution.random(possible_subsplits, concentration, cutoff)
            result.data[clade] = dist
            for subsplit in dist:
                for child in subsplit.nontrivial_children():
                    clade_stack.append(child)
        return result

    @staticmethod
    def from_tree_distribution(tree_distribution):
        result = CCDSet()
        for tree, prob in tree_distribution.items():
            support = tree.to_support()
            for subsplit in support:
                result.add(subsplit, prob)
        result.normalize()
        return result


class SCDSet:
    def __init__(self, cond_probs=None):
        self.data = defaultdict(ProbabilityDistribution)
        self.update(cond_probs)

    def __eq__(self, other):
        if not isinstance(other, SCDSet):
            return False
        if not self.parents() == other.parents():
            return False
        for parent in self.parents():
            if not self[parent] == other[parent]:
                return False
        return True

    def __repr__(self):
        pretty_repr = pprint.pformat(dict(self.data))
        if '\n' in pretty_repr:
            join_str = '\n'
        else:
            join_str = ''
        return join_str.join(("SCDSet(", pretty_repr, ")"))

    def __str__(self):
        strs = [str(parent) + ": " + str(self.get(parent)) for parent in sorted(self.parents())]
        return "\n".join(strs)

    # TODO: Update below here
    def __getitem__(self, item):
        if isinstance(item, Clade):
            return self.data[item]
        if isinstance(item, Subsplit):
            clade = item.clade()
            return self.data[clade][item]

    def add(self, parent, child, probability):
        if not isinstance(parent, Subsplit):
            print("Error: Argument parent not class Subsplit")
            return
        if not isinstance(child, Subsplit):
            print("Error: Argument child not class Subsplit")
            return
        self.data[parent][child] += probability

    def parents(self):
        return self.data.keys()

    def remove(self, parent, child):
        if not isinstance(parent, Subsplit):
            print("Error: Argument parent not class Subsplit")
            return
        if not isinstance(child, Subsplit):
            print("Error: Argument child not class Subsplit")
            return
        self.data[parent].remove(child)

    def set(self, parent, child, probability):
        if not isinstance(parent, Subsplit):
            print("Error: Argument parent not class Subsplit")
            return
        if not isinstance(child, Subsplit):
            print("Error: Argument child not class Subsplit")
            return
        self.data[parent][child] = probability

    def update(self, parent, cond_probs):
        if cond_probs is None:
            return
        if not isinstance(parent, Subsplit):
            print("Error: Argument parent not class Subsplit")
            return
        if not isinstance(cond_probs, collections.Mapping):
            print("Error: Argument cond_probs not a Mapping")
            return
        for subsplit in cond_probs:
            if not isinstance(subsplit, Subsplit):
                print("Warning: Non-Subsplit found in cond_probs, skipping")
                continue
            self.data[parent][subsplit] = cond_probs[subsplit]

    def get(self, item):
        if isinstance(item, Clade):
            return self.data.get(item, ProbabilityDistribution())
        if isinstance(item, Subsplit):
            clade = item.clade()
            return self.data.get(clade, ProbabilityDistribution()).get(item)

    def get_log(self, item):
        prob = self.get(item)

        if prob > 0.0:
            return math.log(prob)
        else:
            return -math.inf

    def check_distributions(self):
        for parent in self.data:
            if not self.data[parent].check():
                return False
        return True

    def bad_distributions(self):
        result = set()
        for parent in self.data:
            if not self.data[parent].check():
                result.add(parent)
        return result

    def normalize(self, parent=None):
        if parent is None:
            for parent in self.data:
                self.data[parent] = self.data[parent].normalize()
        elif isinstance(parent, Subsplit) or isinstance(parent, Clade) and parent == self.root_clade():
            self.data[parent] = self.data[parent].normalize()
        else:
            print("Error: unrecognized argument 'clade'")

    def get_taxon_set(self):
        result = set()
        for parent in self.data:
            if isinstance(parent, Subsplit):
                result |= parent.clade()
        return set().union(*self.data)

    def root_clade(self):
        return Clade(self.get_taxon_set())

    def is_complete(self):
        return self.support().is_complete()

    # TODO: Add PCSSSupport and update this function
    def support(self, include_trivial=False):
        result = SubsplitSupport(include_trivial=include_trivial)
        for clade in self.data:
            # result.add({subsplit for subsplit in self.data[clade] if self.data[clade][subsplit] > 0})
            result.update(self.data[clade].support())
        return result

    def check(self):
        return self.check_distributions() and self.support().is_complete()

    def random_tree(self):
        if not self.is_complete():
            print("Error: CCDSet not complete, no tree possible.")
            return
        root_clade = self.root_clade()
        tree = MyTree()

        clade_stack = [(tree.tree, root_clade)]  # MyTree alteration
        while clade_stack:
            node, clade = clade_stack.pop()
            # possible_subsplits = [subsplit for subsplit in self.data[clade] if not subsplit.is_trivial()]
            # assert len(possible_subsplits) > 0
            assert self.data[clade].check()
            subsplit = self.data[clade].sample()
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

    def tree_distribution(self, verbose=False):
        if not self.is_complete():
            print("Error: CCDSet not complete, no tree possible.")
            return
        root_clade = Clade(self.get_taxon_set())
        tree = MyTree()
        tree.tree.clade = root_clade
        result = TreeDistribution()
        big_stack = [(tree.tree, 0.0)]
        while big_stack:
            current_tree, log_prob_so_far = big_stack.pop()
            if verbose:
                print(f"Current tree: {current_tree}")
                print(f"Current log-probability: {log_prob_so_far}")
            possibilities = []
            clade_stack = [leaf.clade for leaf in current_tree.iter_leaves() if len(leaf.clade) > 1]
            if not clade_stack:
                result.set(MyTree(current_tree), math.exp(log_prob_so_far))
                if verbose:
                    print(f"Sending to output: {current_tree} with probability {math.exp(log_prob_so_far)}")
                continue
            for clade in clade_stack:
                if verbose:
                    print(f"Examining clade: {clade}")
                possibilities.append(subsplit for subsplit, prob in self.data[clade].items()
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
                    new_log_prob += math.log(self.get(subsplit))
                    for child_clade in subsplit.children():
                        child_node = node.add_child(name=str(child_clade))
                        child_node.clade = child_clade
                        if verbose:
                            print(f"Pushing clade: {child_clade}")
                big_stack.append((new_tree, new_log_prob))
                if verbose:
                    print(f"Pushing tree: {new_tree} with probability {math.exp(new_log_prob)}")
        return result

    def clade_probabilities(self):
        result = defaultdict(float)
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, prob_so_far = clade_stack.pop()
            result[clade] += prob_so_far
            cond_dist = self[clade]
            for subsplit in cond_dist:
                new_prob = self[subsplit]
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, prob_so_far * new_prob))
        return result

    # def restrict(self, taxon_set, verbose=False):
    #     taxon_set = Clade(taxon_set)
    #     result = CCDSet()
    #     root_clade = Clade(self.get_taxon_set())
    #     different_stack = [root_clade]
    #     visited_clades = set()  # Testing this out
    #     while different_stack:
    #         base_clade = different_stack.pop()
    #         if base_clade in visited_clades:  # Testing this out
    #             continue
    #         visited_clades.add(base_clade)  # Testing this out
    #         same_stack = [(base_clade, 1.0)]
    #         while same_stack:
    #             clade, base_prob = same_stack.pop()
    #             for subsplit in self[clade]:
    #                 subsplit_prob = self[subsplit]
    #                 clade1, clade2 = subsplit.children()
    #                 restricted_subsplit = subsplit.restrict(taxon_set)
    #                 if restricted_subsplit.is_trivial():
    #                     assert not clade1 & taxon_set or not clade2 & taxon_set
    #                     equivalent_clade = clade1 if (clade1 & taxon_set) else clade2
    #                     same_stack.append((equivalent_clade, base_prob * subsplit_prob))
    #                 else:
    #                     result.add(restricted_subsplit, base_prob * subsplit_prob)
    #                     if len(clade1) > 1:
    #                         different_stack.append(clade1)
    #                     if len(clade2) > 1:
    #                         different_stack.append(clade2)
    #     return result

    def restrict(self, taxon_set, verbose=False):
        taxon_set = Clade(taxon_set)
        result = CCDSet()
        clade_probabilities = self.clade_probabilities()
        clade_stack = [self.root_clade()]
        visited_clades = set()
        for clade, prob in clade_probabilities.items():
            for subsplit in self[clade]:
                subsplit_prob = self[subsplit]
                restricted_subsplit = subsplit.restrict(taxon_set)
                if not restricted_subsplit.is_trivial():
                    result.add(restricted_subsplit, clade_probabilities[clade] * subsplit_prob)
        result.normalize()
        return result

    @staticmethod
    def random(taxon_set, concentration=1.0, cutoff=0.0):
        result = CCDSet()
        root_clade = Clade(taxon_set)
        clade_stack = [root_clade]
        visited_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in visited_clades:
                continue
            visited_clades.add(clade)
            possible_subsplits = list(Subsplit.compatible_subsplits(clade))
            dist = ProbabilityDistribution.random(possible_subsplits, concentration, cutoff)
            result.update(dist)
            for subsplit in dist:
                for child in subsplit.nontrivial_children():
                    clade_stack.append(child)
        return result

    @staticmethod
    def from_tree_distribution(tree_distribution):

        pass

    def iter_clades(self):
        clade_stack = [self.root_clade()]
        visited_clades = set()
        while clade_stack:
            clade = clade_stack.pop()
            if clade in visited_clades:
                continue
            visited_clades.add(clade)
            yield clade

    def iter_subsplits(self):
        for clade in self.iter_clades():
            yield from self.data[clade]

    def unconditional_probabilities(self):
        result = defaultdict(float)
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, clade_prob = clade_stack.pop()
            cond_dist = self[clade]
            for subsplit in cond_dist:
                cond_prob = self[subsplit]
                joint_prob = clade_prob * cond_prob
                result[subsplit] += joint_prob
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, joint_prob))
        return result

    def iter_unconditional_probabilities(self):
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, clade_prob = clade_stack.pop()
            cond_dist = self[clade]
            for subsplit in cond_dist:
                cond_prob = self[subsplit]
                joint_prob = clade_prob * cond_prob
                yield (subsplit, joint_prob)
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, joint_prob))

    def iter_probabilities(self):
        clade_stack = [(self.root_clade(), 1.0)]
        while clade_stack:
            clade, clade_prob = clade_stack.pop()
            cond_dist = self[clade]
            for subsplit in cond_dist:
                cond_prob = self[subsplit]
                joint_prob = clade_prob * cond_prob
                yield (subsplit, cond_prob, joint_prob)
                for child in subsplit.nontrivial_children():
                    clade_stack.append((child, joint_prob))

    def log_likelihood(self, tree, strategy='levelorder'):
        result = 0.0
        for subsplit in tree.traverse_subsplits(strategy):
            cond_prob = self[subsplit]
            if cond_prob > 0.0:
                result += math.log(cond_prob)
            else:
                result += -math.inf
        return result

    def likelihood(self, tree, strategy='levelorder'):
        return math.exp(self.log_likelihood(tree, strategy))

    def kl_divergence_ccd(self, other):
        result = 0.0
        for subsplit, cond_prob, joint_prob in self.iter_probabilities():
            result += -joint_prob * (other.get_log(subsplit) - math.log(cond_prob))
        return result

    def kl_divergence_treedist(self, other):
        result = 0.0
        for tree in other:
            log_q = other.log_likelihood(tree)
            log_p = self.log_likelihood(tree)
            result += -math.exp(log_p) * (log_q - log_p)
        return result

    def kl_divergence(self, other):
        if isinstance(other, CCDSet):
            return self.kl_divergence_ccd(other)
        if isinstance(other, TreeDistribution):
            return self.kl_divergence_treedist(other)
        print("Error: argument 'other' not a CCDNet or TreeDistribution")


def estimate_derivative(prob_of: Subsplit, wrt: Subsplit, ccd: CCDSet, delta: float=0.0001):
    clade = wrt.clade()
    ccd2 = ccd.copy()
    ccd2[clade].add_log(wrt, delta)
    ccd2.normalize(clade)
    uncond = ccd.unconditional_probabilities()
    uncond2 = ccd2.unconditional_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


def estimate_restricted_derivative(restriction, prob_of: Subsplit, wrt: Subsplit, ccd: CCDSet, delta: float=0.0001):
    restriction = Clade(restriction)
    ccd_r = ccd.restrict(restriction)
    clade = wrt.clade()
    ccd2 = ccd.copy()
    ccd2[clade].add_log(wrt, delta)
    ccd2.normalize(clade)
    ccd2_r = ccd2.restrict(restriction)
    uncond = ccd_r.unconditional_probabilities()
    uncond2 = ccd2_r.unconditional_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


def estimate_restricted_clade_derivative(restriction, prob_of: Clade, wrt: Subsplit, ccd: CCDSet, delta: float=0.0001):
    restriction = Clade(restriction)
    ccd_r = ccd.restrict(restriction)
    clade = wrt.clade()
    ccd2 = ccd.copy()
    ccd2[clade].add_log(wrt, delta)
    ccd2.normalize(clade)
    ccd2_r = ccd2.restrict(restriction)
    uncond = ccd_r.clade_probabilities()
    uncond2 = ccd2_r.clade_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


def estimate_restricted_conditional_derivative(restriction, prob_of: Subsplit, wrt: Subsplit, ccd: CCDSet, delta: float=0.0001):
    restriction = Clade(restriction)
    ccd_r = ccd.restrict(restriction)
    clade = wrt.clade()
    ccd2 = ccd.copy()
    ccd2[clade].add_log(wrt, delta)
    ccd2.normalize(clade)
    ccd2_r = ccd2.restrict(restriction)
    est_deriv = (ccd2_r[prob_of] - ccd_r[prob_of]) / delta
    return est_deriv


def estimate_kl_divergence_derivative(ccd: CCDSet, ccd_small: CCDSet, wrt: Subsplit, delta: float=0.0001):
    restriction = ccd_small.root_clade()
    ccd_r = ccd.restrict(restriction)
    clade = wrt.clade()
    ccd2 = ccd.copy()
    ccd2[clade].add_log(wrt, delta)
    ccd2.normalize(clade)
    ccd2_r = ccd2.restrict(restriction)
    est_deriv = (ccd_small.kl_divergence(ccd2_r) - ccd_small.kl_divergence(ccd_r)) / delta
    return est_deriv


def all_restricted_uncond_derivs(ccd: CCDSet, ccd_small: CCDSet, wrt: Subsplit, c2c: dict = None, restricted_self: 'CCDSet' = None):
    restriction = ccd_small.root_clade()
    if not restriction.issubset(ccd.root_clade()):
        raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
    if c2c is None:
        c2c = ccd.clade_to_clade_probabilities()
    if restricted_self is None:
        restricted_self = ccd.restrict(restriction)
    result = dict()
    for sbar in ccd_small.iter_subsplits():
        result[sbar] = ccd.restricted_cond_prob_derivative(restriction, sbar, wrt, c2c, restricted_self)
    return result


def l2norm(arg):
    return np.sqrt(l2norm2(arg))


def l2norm2(arg):
    vals = np.array(list(arg.values()))
    return np.sum(vals**2)


def gradient_descent_one(starting: CCDSet, reference: CCDSet, starting_gamma=1.0, max_iteration=50, verbose=True, const=0.5):
    restriction = reference.root_clade()
    current = starting.copy()
    current_r = current.restrict(restriction)
    current_kl = reference.kl_divergence(current_r)
    current_c2c = current.clade_to_clade_probabilities()
    current_grad = current.restricted_kl_divergence_gradient(other=reference, c2c=current_c2c, restricted_self=current_r)
    current_grad_l2 = l2norm2(current_grad)
    gamma = starting_gamma
    kl_list = [current_kl]
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        candidate = current.copy()
        candidate.add_many_log(current_grad, -gamma)
        candidate.normalize()
        candidate_r = candidate.restrict(restriction)
        candidate_kl = reference.kl_divergence(candidate_r)
        if verbose:
            print(f"Iter {iteration}: KL={candidate_kl:8.4g}")
        # Armijo-Goldstein condition:
        # see https://math.stackexchange.com/questions/373868/optimal-step-size-in-gradient-descent
        if candidate_kl <= current_kl - const*gamma*current_grad_l2:
            current = candidate
            current_r = candidate_r
            current_kl = candidate_kl
            current_c2c = current.clade_to_clade_probabilities()
            current_grad = current.restricted_kl_divergence_gradient(other=reference, c2c=current_c2c, restricted_self=current_r)
            current_grad_l2 = l2norm2(current_grad)
            gamma = gamma * 1.1
        else:
            gamma = gamma * 0.5
            if verbose:
                print(f"Rejected! Shrinking gamma to {gamma}")
        kl_list.append(current_kl)
    if verbose:
        print(f"Final KL={current_kl:8.4g}")
    return current, kl_list


def gradient_descent(starting: CCDSet, references: List[CCDSet], starting_gamma=1.0, max_iteration=50, verbose=True, const=0.5):
    weights = [1] * len(references)
    restrictions = [reference.root_clade() for reference in references]
    current = starting.copy()
    current_rs = [current.restrict(restriction) for restriction in restrictions]
    current_kl = sum(weight*reference.kl_divergence(current_r) for (weight, reference, current_r) in zip(weights, references, current_rs))
    current_c2c = current.clade_to_clade_probabilities()
    current_grad = current.restricted_kl_divergence_gradient_multi(others=references, weights=weights,
                                                                   c2c=current_c2c, restricted_selves=current_rs)
    current_grad_l2 = l2norm2(current_grad)
    gamma = starting_gamma
    kl_list = [current_kl]
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        candidate = current.copy()
        candidate.add_many_log(current_grad, -gamma)
        candidate.normalize()
        candidate_rs = [candidate.restrict(restriction) for restriction in restrictions]
        candidate_kl = sum(weight*reference.kl_divergence(candidate_r) for (weight, reference, candidate_r) in zip(weights, references, candidate_rs))
        if verbose:
            print(f"Iter {iteration}: KL={candidate_kl:8.4g}")
        # Armijo-Goldstein condition:
        # see https://math.stackexchange.com/questions/373868/optimal-step-size-in-gradient-descent
        if candidate_kl <= current_kl - const*gamma*current_grad_l2:
            current = candidate
            current_rs = candidate_rs
            current_kl = candidate_kl
            current_c2c = current.clade_to_clade_probabilities()
            current_grad = current.restricted_kl_divergence_gradient_multi(others=references, weights=weights,
                                                                           c2c=current_c2c, restricted_selves=current_rs)
            current_grad_l2 = l2norm2(current_grad)
            gamma = gamma * 1.1
        else:
            gamma = gamma * 0.5
            if verbose:
                print(f"Rejected! Shrinking gamma to {gamma}")
        kl_list.append(current_kl)
    if verbose:
        print(f"Final KL={current_kl:8.4g}")
    return current, kl_list

