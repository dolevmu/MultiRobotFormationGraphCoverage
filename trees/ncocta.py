from collections import Counter
from math import log, ceil, floor

import numpy as np
from typing import List, Optional
from enum import Enum

from tqdm import tqdm
from treelib import Tree

from trees.configuration import enumerate_config_bottom_up
from trees.traversal import Traversal

class NodeState(Enum):
    UNFINISHED = 1
    INHABITED = 2
    FINISHED = 3

def ncocta_compute_traversal(tree: Tree,
                            num_robots: int,
                            hh: Optional[List[int]] = None) -> Traversal:
    H = tree.depth()  # Tree max depth
    N = max(len(tree.children(v)) for v in tree.nodes) # Tree max degree

    if not hh:
        eps = 1 / (N-1)
        m = floor(log(num_robots, N) - log(log(num_robots, N), N) - 1)
        hm = floor(m + log(m, N) + 5)
        hh = [hm + m - (i + 1) for i in range(m)]

    assert all(h1 >= h2 for h1, h2 in zip([H] + hh, hh))
    robots_lower_bound = 1 + N*hh[0] + sum((N**i - N**(i-1))*hh[i] for i in range(1, len(hh)))
    assert num_robots >= robots_lower_bound, f"Not enough robots: {num_robots} < {robots_lower_bound}"

    current_config = Counter()
    current_config[tree.root] = num_robots
    traversal = [dict(current_config)]

    state_dict = {v: NodeState.UNFINISHED for v in tree.nodes}
    state_dict[tree.root] = NodeState.INHABITED

    counter = 0
    while state_dict[tree.root] != NodeState.FINISHED:
        assert current_config.total() == num_robots, counter
        print(f'{counter}, {current_config}')
        counter += 1

        next_config = Counter()
        for v in enumerate_config_bottom_up(current_config, tree):
            if all(state_dict[u.identifier] == NodeState.FINISHED for u in tree.children(v)):
                state_dict[v] = NodeState.FINISHED
                # There are no robots in T_v subtrees, so we can return to the parent node
                if v != tree.root:
                    # All robots from v go to the parent of v
                    next_config[tree.parent(v).identifier] += current_config[v]
                else:
                    # All robots in v stop
                    next_config[v] += current_config[v]

            elif any(state_dict[u.identifier] != NodeState.FINISHED for u in tree.children(v)):
                # v is unfinished, hence, there are still nodes to explore.
                # We can move to a child.
                if H - tree.depth(v) in hh:
                    # Leave one robot at v
                    next_config[v] += 1
                    # Split the rest equally among the children.
                    per_child = (current_config[v] - 1) // len(tree.children(v))
                    remainder = (current_config[v] - 1) % len(tree.children(v))

                    for u in tree.children(v):
                        if per_child > 0:
                            next_config[u.identifier] += per_child
                            state_dict[u.identifier] = NodeState.INHABITED if tree.children(u.identifier) else NodeState.FINISHED
                    if remainder > 0:
                        next_config[u.identifier] += remainder
                        state_dict[u.identifier] = NodeState.INHABITED if tree.children(u.identifier) else NodeState.FINISHED

                elif H - tree.depth(v) <= hh[0]:
                    # Select a child u of v such that u is unfinished.
                    u = [u.identifier for u in tree.children(v) if state_dict[u.identifier] != NodeState.FINISHED][0]
                    # Move all the robots in v to u leaving one robot in v.
                    next_config[u] += current_config[v] - 1
                    next_config[v] += 1
                else:
                    # Select a child u of v such that u is unfinished.
                    u = [u.identifier for u in tree.children(v) if state_dict[u.identifier] != NodeState.FINISHED][0]
                    # Move all the robots in v to u
                    next_config[u] += current_config[v]
            else:
                # v is inhabited, the subtree is explored but there are still robots in the subtree.
                # The robots wait on a node (vertex) with height in hh (split node) until all the robots arrive.
                # Now, the node is finished and the robots move together to another subtree.
                if H - tree.depth(v) in hh or v == tree.root:
                    # All robots from v remain in v
                    next_config[v] += current_config[v]
                elif tree.parent(v):
                    # All robots from v go to the parent of v leaving one robot in v
                    next_config[tree.parent(v).identifier] += current_config[v] - 1
                    next_config[v] += 1

        # Update config and traversal
        current_config = next_config
        traversal.append(dict(current_config))

    assert current_config.total() == num_robots, counter

    # Return the COCTA traversal
    return tuple(traversal)