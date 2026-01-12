from math import sqrt
from typing import Optional, Tuple, Set

from treelib import Tree

from exploration.baby_giant_step import baby_giant_step
from trees.configuration import Configuration
from trees.traversal import Traversal


def optimistic_bgs(tree: Tree,
                   num_robots: int,
                   start_config: Optional[Configuration] = None) -> Tuple[Traversal, Set[str]]:
    """
    Optimistic BGS:
    - Iteration i: delta = 2^i, can explore depth up to 2^(2i) = delta^2
    - If tree not covered, run with delta = sqrt(k) until finished

    Note: This function modifies the input tree by removing explored nodes.
    """
    sqrt_k = max(1, int(sqrt(num_robots)))
    vertices = set(tree.nodes.keys())
    full_traversal = []
    current_config = start_config

    # Iterations with delta = 2, 4, 8, ... up to sqrt(k)
    i = 1
    while True:
        delta = 2 ** i
        if delta > sqrt_k:
            break
        if len(tree.nodes) <= 1:
            break

        bgs_trav = baby_giant_step(tree, num_robots, max_depth=delta, start_config=current_config)
        full_traversal.extend(list(bgs_trav))

        if full_traversal:
            current_config = full_traversal[-1]

        i += 1

    # If tree not fully covered, run with delta = sqrt(k) until finished
    while len(tree.nodes) > 1:
        bgs_trav = baby_giant_step(tree, num_robots, max_depth=sqrt_k, start_config=current_config)
        if len(bgs_trav) == 0:
            break
        full_traversal.extend(list(bgs_trav))
        if full_traversal:
            current_config = full_traversal[-1]

    deleted = vertices - set(tree.nodes.keys())
    return tuple(full_traversal), deleted
