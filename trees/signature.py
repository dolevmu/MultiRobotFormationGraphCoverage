from collections import Counter, defaultdict
from copy import deepcopy

from frozendict import frozendict
from treelib import Tree
from typing import Tuple, Union, Dict, Set, List

from trees.configuration import Configuration, is_connected, enumerate_configurations, find_root
from trees.transition import is_transition, enumerate_transitions
from trees.traversal import Traversal

UpArrow = '↑'
DownArrow = '↓'
ArrowSymbol = str
FormalConfiguration = Union[Configuration, ArrowSymbol]
FrozenFormalConfiguration = Union[frozendict, ArrowSymbol]
FormalTransition = Tuple[FormalConfiguration, FormalConfiguration]

Signature = Tuple[FormalConfiguration, ...]
FrozenSignature = Tuple[FrozenFormalConfiguration, ...]


def freeze_signature(signature: Signature) -> FrozenSignature:
    return tuple(config if type(config) is str else frozendict(config)
                 for config in signature)


def _project(vertex: str, signature: Signature, tree: Tree) -> Signature:
    assert vertex in tree.nodes, f"Vertex {vertex} not in tree."

    raw_signature = []
    for configuration in signature:
        if type(configuration) is str:
            # if traversal is used as a signature, arrows may change and this should be handled separately
            raw_signature.append(configuration)
        elif configuration.get(vertex, 0) > 0:
            raw_signature.append(configuration)
        elif set(configuration.keys()).issubset(tree.subtree(vertex).nodes):
            configuration_root = find_root(configuration, tree)
            ancestors = {tree.ancestor(configuration_root, level).identifier
                         for level in range(tree.depth(vertex)+1, tree.depth(configuration_root))} | {configuration_root}
            child = ancestors & {child.identifier for child in tree.children(vertex)}
            raw_signature.append(DownArrow + str(child))
        else:
            raw_signature.append(UpArrow)
    return tuple(raw_signature)


def condense(raw_signature: Signature) -> Signature:
    # condense signature:
    condensed_signature = []
    previous_config = None

    for formal_config in raw_signature:
        if formal_config != previous_config:
            condensed_signature.append(formal_config)
            previous_config = formal_config
    return tuple(condensed_signature)


def project(vertex: str, signature: Signature, tree: Tree) -> Signature:
    return condense(_project(vertex, signature, tree))


def is_signature(signature: Signature, vertex: str, tree: Tree) -> bool:
    # Verify vertex is in tree
    assert vertex in tree.nodes, f"Vertex {vertex} not in tree."
    # Verify that there is a common vertex for all configurations
    configurations = [config for config in signature if type(config) is not str]
    assert all(vertex in config for config in configurations), f"Vertex {vertex} must be occupied."
    # Verify configurations are valid
    assert all([is_connected(config, tree) for config in configurations]), f"Configurations must be connected."

    # Verify transitions are correct
    successors = {successor for successor in tree.subtree(vertex).nodes}
    ancestors = {ancestor for ancestor in tree.nodes if ancestor not in tree.subtree(vertex).nodes} | {vertex}
    for config1, config2 in zip(configurations, configurations[1:]):
        if type(config1) is Counter and type(config2) is Counter:
            assert is_transition((config1, config2), tree), f"Invalid transitions found."
        elif type(config1) is Counter and config2 == UpArrow:
            assert set(config1.keys()) & successors == {vertex}, f"To move up, everyone should be at {vertex} or above."
        elif type(config2) is Counter and config1 == UpArrow:
            assert set(config2.keys()) & successors == {vertex}, f"To move up, everyone should be at {vertex} or above."
        elif type(config1) is Counter and config2.startswith(DownArrow):
            assert set(config1.keys()) & ancestors == {vertex}, f"To move down, everyone should be at {vertex} or below."
            assert sum(config1[child.identifier] > 0 for child in tree.children(vertex)) <= 1, f"To move down, only one child may be occupied."
            for child in tree.children(vertex):
                if config1[child.identifier] > 0:
                    assert config2 == DownArrow + str(child.identifier)
        elif type(config2) is Counter and config1.startswith(DownArrow):
            assert set(config2.keys()) & ancestors == {vertex}, f"To move down, everyone should be at {vertex} or below."
            assert sum(config2[child.identifier] > 0 for child in tree.children(vertex)) <= 1, f"To move down, only one child may be occupied."
            for child in tree.children(vertex):
                if config2[child.identifier] > 0:
                    assert config1 == DownArrow + str(child.identifier)
        else:
            assert False, f"Invalid transition: ({config1}, {config2})."

    # Verify no transition repeats
    transitions = [(frozendict(config1), frozendict(config2)) for config1, config2 in zip(signature, signature[1:])
                   if type(config1) is Counter and type(config2) is Counter]
    assert len(set(transitions)) == len(transitions)
    return True


def signatures_precompute(vertex: str, tree: Tree, num_robots: int, raw: bool) -> Dict[FormalConfiguration, Set[FormalConfiguration]]:
    inside_vertex = {v for v in tree.subtree(vertex).nodes}
    outside_vertex = deepcopy(tree)
    outside_vertex.remove_subtree(vertex)
    outside_vertex = {v for v in outside_vertex.nodes}

    # Pre-computation: get all configurations that occupy vertex
    valid_configurations = enumerate_configurations(vertex, tree, num_robots)

    # Pre-computation: for each configuration, get list of valid transitions
    valid_transitions = defaultdict(lambda: set())

    for configuration in valid_configurations:
        collected_transitions = enumerate_transitions(configuration, tree)

        for second_configuration in collected_transitions:
            # Handle ↑
            if set(second_configuration.keys()).issubset(outside_vertex):
                valid_transitions[frozendict(configuration)].add(UpArrow)
                valid_transitions[UpArrow].add(frozendict(configuration))
            # Handle ↓
            elif set(second_configuration.keys()).issubset(inside_vertex - {vertex}):
                # If raw, explicitly store raw second_configuration, otherwise store symbolic ↓
                second_configuration = frozendict(second_configuration) if raw else DownArrow
                valid_transitions[frozendict(configuration)].add(frozendict(second_configuration))
                # Add the other direction as second_configuration won't be enumerated
                valid_transitions[second_configuration].add(frozendict(configuration))
            else:
                valid_transitions[frozendict(configuration)].add(frozendict(second_configuration))
                # The other direction will also be covered when reaching configuration=second_configuration

    return valid_transitions


def enumerate_signatures(vertex: str, tree: Tree, num_robots: int, raw: bool) -> List[Signature]:
    #  Let G=(V,E) be the following graph:
    #  1. V = {Configurations that occupy vertex} + {'↑'} + {Configurations projected to '↓'} (raw=True)
    #                                                     + {'↓'} (raw=False)
    #  2. E = an edge exists iff it is a valid transition
    # G is represented with an adjacency list valid_transitions:
    valid_transitions = signatures_precompute(vertex, tree, num_robots, raw=raw)

    collected_signatures = []
    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    # We need to enumerate all possible paths in G that don't repeat an edge.
    # This corresponds to signatures where a transition does not repeat.
    def dfs_scan_signatures(current_signature: List[FormalConfiguration],
                            used_transitions: Dict[FormalConfiguration, Set[FormalConfiguration]]):
        if len(tree.children(vertex)) == 0 and sum(type(config) is not str for config in current_signature) > 1:
            # If vertex is a leaf, we can assume w.l.o.g that it is visited precisely once.
            # Indeed, connected configurations are collapsible.
            return

        if any(type(config) is not str for config in current_signature if type(config) is not str):
            collected_signatures.append(current_signature)
        for next_config in valid_transitions[current_signature[-1]] - used_transitions[current_signature[-1]]:
            # Check if transition was used
            if next_config in used_transitions[current_signature[-1]]:
                continue
            used_transitions[current_signature[-1]].add(next_config)
            next_signature = current_signature+[next_config]
            dfs_scan_signatures(next_signature, deepcopy(used_transitions))

    dfs_scan_signatures([start_config], used_transitions=defaultdict(lambda: set()))
    return collected_signatures

