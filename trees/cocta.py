from collections import Counter
from math import log, ceil, floor

import numpy as np
from typing import List, Optional
from enum import Enum

from tqdm import tqdm
from treelib import Tree

from trees.configuration import enumerate_config_bottom_up, find_root, Configuration
from trees.ncocta import H_TABLE
from trees.table import fpt_compute_traversal
from trees.traversal import Traversal
from trees.tree import print_tree


class NodeState(Enum):
    UNFINISHED = 1
    INHABITED = 2
    FINISHED = 3

def cocta_compute_traversal(tree: Tree,
                            num_robots: int,
                            hh: Optional[List[int]] = None,
                            start_config: Optional[Configuration] = None) -> Traversal:
    if num_robots == 1:
        return fpt_compute_traversal(tree, 1)

    H = tree.depth()  # Tree max depth
    N = max(len(tree.children(v)) for v in tree.nodes) # Tree max degree

    if not hh:
        if num_robots in H_TABLE:
            hh = H_TABLE[num_robots]
        else:
            m = ceil(log(num_robots, N) - log(log(num_robots, N), N) - 1)
            hm = floor(m + log(m, N) + 5)
            hh = [hm + m - (i + 1) for i in range(m)]

    if not start_config:
        current_config = Counter()
        current_config[tree.root] = num_robots
    else:
        current_config = start_config
    root = find_root(current_config, tree)
    traversal = [dict(current_config)]

    state_dict = {v: NodeState.UNFINISHED for v in tree.nodes}
    state_dict[tree.root] = NodeState.INHABITED

    counter = 1
    while any(state_dict[u] == NodeState.UNFINISHED for u in tree.expand_tree(root)):
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
                to_explore = 1 + sum(len(list(tree.expand_tree(u)))
                                     for u in tree.expand_tree(v) if state_dict[u] != NodeState.FINISHED)
                available_robots = current_config[v] - 1
                if current_config[v] > to_explore:
                    available_robots = to_explore - 1  # The rest moved to the parent or stay at v
                    if tree.parent(v):
                        next_config[tree.parent(v).identifier] += current_config[v] - to_explore
                    else:
                        next_config[v] += current_config[v] - to_explore

                if H - tree.depth(v) in hh:
                    # Leave one robot at v
                    next_config[v] += 1

                    # Split the rest equally among the children.
                    unfinished = [u.identifier for u in tree.children(v)
                                  if state_dict[u.identifier] != NodeState.FINISHED]
                    per_child = available_robots // len(unfinished)
                    remainder = available_robots % len(unfinished)

                    for u in unfinished:
                        if per_child > 0:
                            next_config[u] += per_child
                            state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED
                    if remainder > 0:
                        next_config[u] += remainder
                        state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED

                elif H - tree.depth(v) <= hh[0]:
                    # Leave one robot at v
                    next_config[v] += 1

                    # Select a child u of v such that u is unfinished.
                    unfinished_childs = [u.identifier for u in tree.children(v) if state_dict[u.identifier] != NodeState.FINISHED]
                    to_explore = np.cumsum([len(list(tree.expand_tree(u))) for u in unfinished_childs])
                    childs_to_explore = sum(available_robots >= to_explore)
                    # Move all robots in v to u leaving one robot in v.
                    # If there are more robots than nodes to explore, select another child to move
                    # the rest to.
                    for u in unfinished_childs[:childs_to_explore]:
                        next_config[u] += len(list(tree.expand_tree(u)))
                        state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED
                    if childs_to_explore > 0:
                        remainder = available_robots - int(to_explore[childs_to_explore-1])
                    else:
                        remainder = available_robots
                    if len(unfinished_childs) > childs_to_explore and remainder > 0:
                        next_config[unfinished_childs[childs_to_explore]] += remainder
                        state_dict[unfinished_childs[childs_to_explore]] = NodeState.INHABITED if tree.children(unfinished_childs[childs_to_explore]) else NodeState.FINISHED
                    elif tree.parent(v) and remainder > 0:
                        next_config[tree.parent(v).identifier] += remainder
                    else:
                        next_config[v] += remainder
                else:
                    # Select a child u of v such that u is unfinished.
                    unfinished_childs = [u.identifier for u in tree.children(v) if state_dict[u.identifier] != NodeState.FINISHED]

                    # If child subtree has more nodes to explore than available robots, move all to child
                    u = unfinished_childs[0]
                    to_explore = len(list(tree.expand_tree(u)))
                    can_jump = not tree.parent(u) or current_config[tree.parent(u)] == 0
                    if to_explore >= current_config[v] and can_jump:
                        next_config[u] += current_config[v]
                        state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED
                    else:
                        # If there are more robots than nodes to explore, leave one robot in v and select another child to move
                        # the rest to.
                        next_config[v] += 1

                        to_explore = np.cumsum([len(list(tree.expand_tree(u))) for u in unfinished_childs])
                        childs_to_explore = sum(available_robots >= to_explore)
                        # Move all robots in v to u leaving one robot in v.
                        # If there are more robots than nodes to explore, select another child to move
                        # the rest to.
                        for u in unfinished_childs[:childs_to_explore]:
                            next_config[u] += len(list(tree.expand_tree(u)))
                            state_dict[u] = NodeState.INHABITED if tree.children(u) else NodeState.FINISHED
                        if childs_to_explore > 0:
                            remainder = available_robots - int(to_explore[childs_to_explore - 1])
                        else:
                            remainder = available_robots - 1
                        if len(unfinished_childs) > childs_to_explore and remainder > 0:
                            next_config[unfinished_childs[childs_to_explore]] += remainder
                            state_dict[unfinished_childs[childs_to_explore]] = NodeState.INHABITED if tree.children(unfinished_childs[childs_to_explore]) else NodeState.FINISHED
                        elif tree.parent(v) and remainder > 0:
                            next_config[tree.parent(v).identifier] += remainder
                        else:
                            next_config[v] += remainder
            else:
                # v is inhabited, the subtree is explored but there are still robots in the subtree.
                # The robots leave one robot behind to maintain communication and drip to the parent node.
                if tree.parent(v):
                    next_config[tree.parent(v).identifier] += current_config[v] - 1
                    next_config[v] += 1
                else:
                    next_config[v] += current_config[v]

        if next_config == current_config:
            print_tree(tree.subtree(find_root(current_config, tree)))
            assert next_config != current_config, f"Not enough robots, stuck at ({counter}) {current_config}."


        # Update config and traversal
        current_config = next_config
        traversal.append(dict(current_config))

        assert sum(current_config.values()) == num_robots, counter
        print(f'{counter}, {current_config}')
        counter += 1

    print(f'{counter}, {current_config}')
    counter += 1

    # Return the COCTA traversal
    return tuple(traversal)