from collections import Counter
from itertools import combinations, product
import msgpack
import numpy as np
from typing import Tuple, List, Optional, Union, Dict

from frozendict import frozendict
from treelib import Tree

Configuration = Dict[str, int]  # Maps vertex id to number of occupying robots

UpArrow = '↑'
DownArrow = '↓'
ArrowSymbol = str
FormalConfiguration = Union[Configuration, ArrowSymbol]
FrozenFormalConfiguration = Union[frozendict, ArrowSymbol]
FormalTransition = Tuple[FormalConfiguration, FormalConfiguration]

def find_root(configuration: Configuration, tree: Tree) -> Optional[str]:
    # Find the root node of the configuration
    configuration_root = None
    for vertex in configuration:
            if tree.parent(vertex) is None or tree.parent(vertex).identifier not in configuration:
                if configuration_root is not None:
                    return None  # Multiple roots found, not a single subtree
                configuration_root = vertex
    return configuration_root


def is_connected(configuration: Configuration, tree: Tree) -> bool:
    # check configuration states only
    assert all([num_robots > 0 for num_robots in configuration.values()]), f"Some vertices in {configuration} are not occupied."
    # check all occupied vertices are part of the tree
    assert all([vertex in tree.nodes for vertex in configuration]), f"Some vertices in {configuration} are not in tree."

    # Find the root node of the configuration
    configuration_root = find_root(configuration, tree)
    if configuration_root is None:
        return False  # No root node found

    # Traverse the subtree starting from configuration_root
    subtree_vertices = {configuration_root}
    nodes_to_visit = [configuration_root]

    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        for child in tree.children(current_node):
            if child.identifier in configuration:
                subtree_vertices.add(child.identifier)
                nodes_to_visit.append(child.identifier)

    # Check if the subtree matches the subset
    return subtree_vertices == set(configuration.keys())


# Recursively copy nodes from the parent subtree, excluding the vertex (current child)
def add_subtree_excluding(tree, current_vertex, exclude_vertex, new_tree, tail):
    for child in tree.children(current_vertex):
        if child.identifier != exclude_vertex:
            on_path = tree.is_ancestor(child.identifier, exclude_vertex)
            if tail > 0 or on_path:
                new_tree.create_node(child.tag, child.identifier, parent=current_vertex)
                add_subtree_excluding(tree, child.identifier, exclude_vertex, new_tree, tail+on_path)


def enumerate_configurations(vertex: str, tree: Tree, num_robots: int) -> List[Configuration]:
    assert vertex in tree.nodes, f"Vertex {vertex} is not in the tree."

    if num_robots == 0:
        return [Counter({})]

    subtrees = [(child.identifier, tree.subtree(child.identifier)) for child in tree.children(vertex)]

    # Handle the case where the vertex has a parent
    parent = tree.parent(vertex)
    node_depth = tree.depth(vertex)
    if node_depth > 0 and num_robots > 1:
        # parent_tree = deepcopy(tree)
        # parent_tree.remove_node(vertex)
        # subtrees = [(tree.parent(vertex).identifier, parent_tree)] + subtrees

        target_depth = max(0, node_depth - num_robots + 1)
        # Get the ancestor at the target depth
        k_ancesstor = tree.ancestor(vertex, level=target_depth)
        k_ancesstor_subtree = tree.subtree(k_ancesstor.identifier)
        # Create a new subtree for the parent that excludes the current vertex (child)
        new_k_ancesstor_subtree = Tree()
        new_k_ancesstor_subtree.create_node(k_ancesstor.tag, k_ancesstor.identifier)  # Add the root of the parent
        # Add the rest of the parent subtree, excluding the current vertex (child)
        tail = max(0, num_robots - 1 - node_depth)
        add_subtree_excluding(k_ancesstor_subtree, k_ancesstor.identifier, vertex, new_k_ancesstor_subtree, tail)

        subtrees = [(parent.identifier, new_k_ancesstor_subtree)] + subtrees

    # The vertex has itself, its parent (if exists), and its children, at its neighborhood
    num_neighbors = 1 + len(subtrees)

    collected_configurations = list()

    # Enumerate on how many robots are at the vertex, outside of vertex's subtree, and at each of its children subtrees
    for separators in combinations(range(num_robots+num_neighbors-1), num_neighbors-1):
        separators = [-1] + list(separators) + [num_robots+num_neighbors-1]
        robot_counts = np.diff(separators)-1
        assert sum(robot_counts) == num_robots
        if robot_counts[0] == 0:
            continue

        options = product(*[enumerate_configurations(neighbor, subtree, robot_counts[i])
                           for i, (neighbor, subtree) in enumerate(subtrees, 1)])
        for option in options:
            configuration = Counter({vertex: int(robot_counts[0])})
            for config in option:
                configuration.update(config)
            collected_configurations.append(configuration)

    del subtrees  # Delete the new parent tree

    return collected_configurations


def split_configuration(configuration: Configuration, tree: Tree) -> Tuple[Configuration, Configuration]:
    assert is_connected(configuration, tree), "Configuration is invalid"

    num_robots = sum(configuration.values())

    configuration_root = find_root(configuration, tree)

    # DFS scan until finding a sub-configuration of size > num_robots / 2
    subtree_counter = Counter({})

    def dfs(vertex, threshold) -> Optional[int]:
        children = {child.identifier for child in tree.children(vertex) if child.identifier in configuration}
        size = configuration[vertex]
        for child in children:
            res = dfs(child, threshold)
            if res:
                return res
            size += subtree_counter[child]
        if vertex == configuration_root:
            # Pick child of largest size
            return subtree_counter.most_common(1)[0][0]
        subtree_counter.update({vertex: size})
        if size >= threshold:
            # TODO: suboptimal, may find a partition closer to threshold
            return vertex

    split_vertex = dfs(configuration_root, threshold=round(num_robots / 2))

    configuration_child = Counter({v: k for v, k in configuration.items() if v in tree.subtree(split_vertex)})
    configuration_parent = Counter({v: k for v, k in configuration.items() if v not in tree.subtree(split_vertex)})

    return configuration_child, configuration_parent


def pack_configuration(configuration: FormalConfiguration):
    if type(configuration) is str:
        return configuration
    return msgpack.packb(tuple(sorted(configuration.items())))


def unpack_configuration(packed_config):
    if type(packed_config) is str:
        return packed_config
    return frozendict({v: count for v, count in msgpack.unpackb(packed_config)})