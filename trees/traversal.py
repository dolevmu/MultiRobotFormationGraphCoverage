from typing import Tuple
from treelib import Tree

from trees.configuration import Configuration, is_connected
from trees.transition import is_transition

Traversal = Tuple[Configuration, ...]

def is_traversal(traversal: Traversal, tree: Tree) -> bool:
    for configuration in traversal:
        if not is_connected(configuration, tree):
            return False  #  configuration is invalid

    for transition in zip(traversal, traversal[1:]):
        if not is_transition(transition, tree):
            return False  # transition is invalid

    # validate coverage
    covered_vertices = {vertex for configuration in traversal for vertex in configuration}
    tree_vertices = set(tree.nodes.keys())
    if covered_vertices != tree_vertices:
        return False  # tree is not covered
    return True


def print_traversal(traversal: Traversal, chunk_size=4):
    for t in range(0, len(traversal), chunk_size):
        chunk = traversal[t:t + chunk_size]
        print(chunk)
