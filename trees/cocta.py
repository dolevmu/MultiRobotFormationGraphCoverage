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

def cocta_compute_traversal(tree: Tree,
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
        assert num_robots >= 1 + N*hh[0] + sum((N**i - N**(i-1))*hh[i] for i in range(1, len(hh)))

    current_config = Counter()
    current_config[tree.root] = num_robots
    traversal = [dict(current_config)]

    state_dict = {v: NodeState.UNFINISHED for v in tree.nodes}
    state_dict[tree.root] = NodeState.INHABITED

    counter = 0
    while state_dict[tree.root] != NodeState.FINISHED:
        assert current_config.total() == num_robots, counter
        print(f'{counter}, {current_config}')
        if counter == 71:
            print('here')
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
                # If there are more robots than nodes to explore, we can use the rest of the robots
                # to explore a different subtree.
                if H - tree.depth(v) in hh:
                    # If there are more robots than nodes to explore, move the rest of the
                    # robots to the parent.
                    to_explore = sum(state_dict[u] != NodeState.FINISHED
                                     for u in tree.expand_tree(v))
                    if to_explore < current_config[v]:
                        next_config[tree.parent(v).identifier] += current_config[v] - to_explore
                        # Split the rest equally among the children.
                        per_child = (to_explore - 1) // len(tree.children(v))
                        remainder = (to_explore - 1) % len(tree.children(v))
                    else:
                        per_child = (current_config[v] - 1) // len(tree.children(v))
                        remainder = (current_config[v] - 1) % len(tree.children(v))

                    for u in tree.children(v):
                        if per_child > 0:
                            next_config[u.identifier] += per_child
                            state_dict[u.identifier] = NodeState.INHABITED if tree.children(u.identifier) else NodeState.FINISHED
                    if remainder > 0:
                        next_config[u.identifier] += remainder
                    # Leave one robot at v
                    next_config[v] += 1
                elif H - tree.depth(v) <= hh[0]:
                    # Select a child u of v such that u is unfinished.
                    unfinished_childs = [u.identifier for u in tree.children(v) if state_dict[u.identifier] != NodeState.FINISHED]
                    to_explore = np.cumsum([len(list(tree.expand_tree(u))) for u in unfinished_childs])
                    childs_to_explore = sum(current_config[v] - 1 >= to_explore)
                    # Move all robots in v to u leaving one robot in v.
                    # If there are more robots than nodes to explore, select another child to move
                    # the rest to.
                    for u in unfinished_childs[:childs_to_explore]:
                        next_config[u] += len(list(tree.expand_tree(u)))
                        state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED

                    if childs_to_explore > 0:
                        remainder = current_config[v] - 1 - int(to_explore[childs_to_explore-1])
                    else:
                        remainder = current_config[v] - 1
                    if len(unfinished_childs) > childs_to_explore and remainder > 0:
                        next_config[unfinished_childs[childs_to_explore]] += remainder
                        next_config[v] += 1
                    elif tree.parent(v) and remainder > 0:
                        next_config[tree.parent(v).identifier] += remainder
                        next_config[v] += 1
                    else:
                        next_config[v] += remainder
                        next_config[v] += 1
                else:
                    # Select a child u of v such that u is unfinished.
                    unfinished_childs = [u.identifier for u in tree.children(v) if state_dict[u.identifier] != NodeState.FINISHED]
                    to_explore = np.cumsum([len(list(tree.expand_tree(u))) for u in unfinished_childs])
                    # If there are more nodes to explore than robots, move all robots in v to u.
                    if to_explore[-1] >= current_config[v] and current_config[v] == num_robots:
                        next_config[unfinished_childs[0]] += current_config[v]
                        state_dict[unfinished_childs[0]] = NodeState.INHABITED if tree.children(unfinished_childs[0]) else NodeState.FINISHED
                    elif to_explore[-1] >= current_config[v]:
                        next_config[v] += current_config[v]
                    elif to_explore[-1] < current_config[v]:
                        # If there are more robots than nodes to explore, leave one robot in v and select another child to move
                        # the rest to.
                        next_config[v] += 1
                        for u in unfinished_childs:
                            next_config[u] += len(list(tree.expand_tree(u)))
                            state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED
                        remainder = current_config[v] - 1 - int(to_explore[-1])
                        if remainder > 0:
                            if tree.parent(v):
                                next_config[tree.parent(v).identifier] += remainder
                            else:
                                next_config[v] += remainder
                    # else:
                    #     childs_to_explore = sum(current_config[v] - 1 >= to_explore)
                    #     for u in unfinished_childs[:childs_to_explore]:
                    #         next_config[u] += len(list(tree.expand_tree(u)))
                    #         state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED
                    #     if childs_to_explore == 0:
                    #         remaining_robots = current_config[v] - 1
                    #     else:
                    #         remaining_robots = current_config[v] - 1 - int(to_explore[childs_to_explore-1])
                    #     if remaining_robots > 0:
                    #         if len(unfinished_childs) > childs_to_explore:
                    #             next_config[unfinished_childs[childs_to_explore]] += remaining_robots
                    #             state_dict[unfinished_childs[childs_to_explore]] = NodeState.INHABITED if tree.children(unfinished_childs[childs_to_explore]) else NodeState.FINISHED
                    #         elif tree.parent(v):
                    #             next_config[tree.parent(v).identifier] += remaining_robots
                    #         else:
                    #             next_config[v] += remaining_robots
                    #     next_config[v] += 1
            else:
                # v is inhabited, the subtree is explored but there are still robots in the subtree.
                # The robots leave one robot behind to maintain communication and drip to the parent node.
                if tree.parent(v):
                    next_config[tree.parent(v).identifier] += current_config[v] - 1
                    next_config[v] += 1

        # Update config and traversal
        current_config = next_config
        traversal.append(dict(current_config))

    assert current_config.total() == num_robots, counter

    # Return the COCTA traversal
    return tuple(traversal)