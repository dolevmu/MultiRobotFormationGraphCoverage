import gc
from collections import Counter
from typing import Iterator, List
from copy import copy, deepcopy

from frozendict import frozendict
from treelib import Tree

from trees.configuration import UpArrow, FormalConfiguration, Configuration, enumerate_configurations
from trees.signature import Signature


def cocta_enumerate_initial_configuration(vertex: str, tree: Tree, num_robots: int) -> List[Configuration]:
    # Returns an enumerator for all possible ways of getting into a subtree
    assert vertex in tree.nodes, f"Vertex {vertex} is not in the tree."

    # Not memory optimal
    parent_tree = deepcopy(tree)
    for u in tree.children(vertex):
        parent_tree.remove_node(u.identifier)

    collected_configurations = enumerate_configurations(vertex, parent_tree, num_robots)

    del parent_tree
    gc.collect()

    return collected_configurations


def cocta_scan_signatures(current_signature: Signature, tree: Tree):
    num_robots = sum(current_signature[-1].values())


def enumerate_cocta_signatures(vertex: str,
                               tree: Tree,
                               num_robots: int) -> Iterator[Signature]:
    # First, we want to enumerate all possible starting configurations.
    starting_configs = enumerate_cocta_signatures(vertex, tree, num_robots)
    current_signature = [UpArrow] if tree.parent(vertex) else []

    for start_config in starting_configs:
        next_signature = current_signature[:]+[start_config]
        yield from cocta_scan_signatures(next_signature)

