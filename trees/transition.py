from collections import Counter
from frozendict import frozendict
from itertools import combinations, product
import numpy as np
from typing import Tuple, List, Set

from pulp import PULP_CBC_CMD
from treelib import Tree
import pulp

from trees.configuration import Configuration, is_connected, split_configuration, find_root, UpArrow, DownArrow, \
    FormalTransition

Transition = Tuple[Configuration, Configuration]  # A pair of consecutive configurations


def solve_ilp(transition: Transition, tree: Tree) -> bool:
    # Create the problem instance
    problem = pulp.LpProblem("RobotTransition", pulp.LpMinimize)

    # Create variables for the number of robots moving along each edge
    tree_vertices = list(set(transition[0].keys()) | set(transition[1].keys()))
    edges = [(u, v) for u in tree_vertices for v in tree.is_branch(u) if v in tree_vertices]

    e_var = {}
    for (u, v) in edges:
        e_var[(u, v)] = pulp.LpVariable(f"e[{u},{v}]",
                                        upBound=transition[0].get(u, 0),  # move all robots from u to v
                                        lowBound=0,  # get all robots from v to u
                                        cat='Integer')
        e_var[(v, u)] = pulp.LpVariable(f"e[{v},{u}]",
                                        upBound=transition[0].get(v, 0),  # move all robots from v to u
                                        lowBound=0,  # get all robots from v to u
                                        cat='Integer')

    # Target satisfaction constraints
    for v in tree_vertices:
        to_children = pulp.lpSum(e_var[(v, u.identifier)] for u in tree.children(v) if (v, u.identifier) in edges)
        from_children = pulp.lpSum(e_var[(u.identifier, v)] for u in tree.children(v) if (v, u.identifier) in edges)
        to_parent = 0
        from_parent = 0
        if tree.parent(v):
            to_parent = e_var.get((v, tree.parent(v).identifier), 0)
            from_parent = e_var.get((tree.parent(v).identifier, v), 0)
        problem += (from_children + from_parent + transition[0].get(v, 0) ==
                    to_children + to_parent + transition[1].get(v, 0)), f"FlowSatisfaction_{v}"

    # Solve
    problem.solve(PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[problem.status]

    return status in ['Optimal', 'Integer Feasible']

def is_transition(transition: Transition, tree: Tree) -> bool:
    assert is_connected(transition[0], tree), f"Current configuration {transition[0]} is invalid."
    assert is_connected(transition[1], tree), f"Target configuration {transition[1]} is invalid."
    assert sum(transition[0].values()) == sum(transition[1].values()), f"Number of robots is not preserved."

    # First, we can restrict tree to the nodes occupied in the transition
    tree_vertices = list(set(transition[0].keys()) | set(transition[1].keys()))
    # Second, this must form a connected subtree
    if not is_connected(Counter({vertex: 1 for vertex in tree_vertices}), tree):
        return False  # current and target configurations are not connected

    # Setup and solve the ILP
    return solve_ilp(transition, tree)


def is_up_transition(vertex: str, transition: FormalTransition, tree: Tree, valid_transitions: Set[Configuration]) -> bool:
    if transition[0] == UpArrow:
        return True

    if UpArrow in valid_transitions:
        return transition[1] == UpArrow

    if type(transition[0]) is str and transition[0].startswith(DownArrow):
        return True

    # check that we only go up
    for v in set(transition[0].keys()) & tree.subtree(vertex):
        expected = sum(transition[0][child.identifier] for child in tree.children(v))
        if transition[1][v] != expected:
            return False

    return True


def _enumerate_transitions(configuration: Configuration, tree: Tree) -> List[Configuration]:
    assert is_connected(configuration, tree), "Invalid input configuration."
    num_occupied_vertices = len(configuration.keys())
    num_robots = configuration.total()

    if num_occupied_vertices == 1:
        vertex = list(configuration.keys())[0]
        # The vertex has itself, its parent (if exists), and its children, at its neighborhood
        neighbors = [child.identifier for child in tree.children(vertex)]
        if tree.parent(vertex):
            neighbors = [tree.parent(vertex).identifier] + neighbors
        neighbors = [vertex] + neighbors

        collected_configurations = list()

        # Enumerate on how many robots are at the vertex, outside of vertex's subtree, and at each of its children
        # subtrees
        for separators in combinations(range(num_robots + len(neighbors) - 1), len(neighbors) - 1):
            separators = [-1] + list(separators) + [num_robots + len(neighbors) - 1]
            robot_counts = np.diff(separators) - 1
            collected_configurations.append(Counter({v: int(k) for v, k in zip(neighbors, robot_counts) if k > 0}))

        # Note that these may not be connected, but the other part of the configuration may cover for that...
        return collected_configurations

    # Split configuration into two
    configuration_child, configuration_parent = split_configuration(configuration, tree)
    split = find_root(configuration_child, tree)
    parent_split = tree.parent(split).identifier

    # Get the set of transitions for each of the configuration parts
    child_collected_configurations = _enumerate_transitions(configuration_child, tree)
    parent_collected_configurations = _enumerate_transitions(configuration_parent, tree)

    collected_configurations = list()

    # Now we need to merge the two parts and keep connectivity at split
    # Option 1: Parent config (2) moves down to occupy split, and child config (1) moves down
    parent_collected_configurations_occupied_split = [parent_config for parent_config in parent_collected_configurations
                                                      if parent_config[split] > 0]
    child_collected_configurations_unoccupied_split = [child_config for child_config in child_collected_configurations
                                                       if
                                                       child_config[split] == 0]
    for child_config, parent_config in product(child_collected_configurations_unoccupied_split,
                                               parent_collected_configurations_occupied_split):
        config = Counter()
        config.update(child_config)
        config.update(parent_config)
        # TODO: maybe we can strike out some options here...
        collected_configurations.append(config)

    # Option 2: Child config (1) moves up to occupy parent of split, and parent config (2) moves up / down
    child_collected_configurations_occupied_parent_split = [child_config for child_config in child_collected_configurations
                                                            if child_config[parent_split] > 0 and is_connected(child_config, tree)]
    parent_collected_configurations_unoccupied_parent_split = [parent_config for parent_config in parent_collected_configurations
                                                               if parent_config[parent_split] == 0 and parent_config[split] == 0]  # ignore robot swaps
    for child_config, parent_config in product(child_collected_configurations_occupied_parent_split,
                                               parent_collected_configurations_unoccupied_parent_split):
        config = Counter()
        config.update(child_config)
        config.update(parent_config)
        # TODO: maybe we can strike out some options here...
        collected_configurations.append(config)

    # Option 3: Child config stays on split and keeps connectivity, and parent config is connected to split or its parent
    child_collected_configurations_occupied_split = [child_config for child_config in child_collected_configurations
                                                     if child_config[split] > 0 and is_connected(child_config, tree)]
    parent_collected_configurations_occupied_parent_split = [parent_config for parent_config in
                                                             parent_collected_configurations
                                                             if parent_config[parent_split] > 0 or parent_config[
                                                                 split] > 0]
    for child_config, parent_config in product(child_collected_configurations_occupied_split,
                                               parent_collected_configurations_occupied_parent_split):
        config = Counter()
        config.update(child_config)
        config.update(parent_config)
        # TODO: maybe we can strike out some options here...
        collected_configurations.append(config)

    # Option 4: split is not occupied.
    # If split is not occupied in both configurations, then we can only keep connectivity by moving to parent
    child_collected_configurations_unoccupied_split = [child_config for child_config in child_collected_configurations
                                                       if child_config[parent_split] == child_config.total()]
    parent_collected_configurations_unoccupied_split = [parent_config for parent_config in
                                                        parent_collected_configurations
                                                        if parent_config[split] == 0]
    for child_config, parent_config in product(child_collected_configurations_unoccupied_split,
                                               parent_collected_configurations_unoccupied_split):
        config = Counter()
        config.update(child_config)
        config.update(parent_config)
        collected_configurations.append(config)

    # remove duplicates
    collected_unique_configurations = {frozendict(configuration) for configuration in collected_configurations}
    return [Counter(frozen_config) for frozen_config in collected_unique_configurations]


def enumerate_transitions(configuration: Configuration, tree: Tree) -> List[Configuration]:
    # Keep only connected configurations that are different from input configuration
    return [config for config in _enumerate_transitions(configuration, tree)
            if config != configuration and is_connected(config, tree)]
