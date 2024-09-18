from collections import Counter, defaultdict
from copy import deepcopy

from frozendict import frozendict
from treelib import Tree
from typing import Tuple, Union, Dict, Set, List

from trees.configuration import Configuration, is_connected, enumerate_configurations, find_root
from trees.transition import is_transition, enumerate_transitions

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
            child = list(ancestors & {child.identifier for child in tree.children(vertex)})
            assert len(child) == 1
            raw_signature.append(DownArrow + child[0])
        else:
            raw_signature.append(UpArrow)
    return tuple(raw_signature)


def condense(raw_signature: Signature) -> Signature:
    # condense signature:
    condensed_signature = []
    previous_config = None

    for formal_config in raw_signature:
        if type(formal_config) is str and type(previous_config) is str:
            if formal_config[0] == DownArrow and previous_config[0] == DownArrow:
                continue
        if formal_config != previous_config:
            condensed_signature.append(formal_config)
            previous_config = formal_config
    return tuple(condensed_signature)


def project(vertex: str, signature: Signature, tree: Tree) -> Signature:
    return condense(_project(vertex, signature, tree))


def get_child_key(vertex: str, child: str, signature: Signature, tree: Tree) -> Signature:
    assert any(child == v.identifier for v in tree.children(vertex)), f"{child} is not a child of {vertex} in tree."

    # Handle the down arrows
    sig = []
    for config in signature:
        if type(config) is str:
            if config in [UpArrow, DownArrow + child]:
                sig.append(config)
            else:
                # A brother is upper to child
                sig.append(UpArrow)
        else:
            sig.append(deepcopy(config))

    return project(child, tuple(sig), tree)


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
    # TODO: this allows ↑,↓ transitions to repeat, we can avoid that with valid_transitions.
    return True


def signatures_precompute(vertex: str, tree: Tree, num_robots: int, raw: bool
                          ) -> Tuple[Dict[FormalConfiguration, Set[FormalConfiguration]], Dict[FormalTransition, int]]:
    inside_vertex = {v for v in tree.subtree(vertex).nodes}
    outside_vertex = deepcopy(tree)
    outside_vertex.remove_subtree(vertex)
    outside_vertex = {v for v in outside_vertex.nodes}

    # Pre-computation: get all configurations that occupy vertex
    valid_configurations = enumerate_configurations(vertex, tree, num_robots)

    # Pre-computation: for each configuration, get list of valid transitions
    valid_transitions = defaultdict(lambda: set())

    # Pre-computation: for each configuration, get the budget for each arrow: how many such transitions are available
    budget = defaultdict(lambda: 0)

    for configuration in valid_configurations:
        collected_transitions = enumerate_transitions(configuration, tree)

        for second_configuration in collected_transitions:
            # Handle ↑
            if set(second_configuration.keys()).issubset(outside_vertex):
                valid_transitions[frozendict(configuration)].add(UpArrow)
                valid_transitions[UpArrow].add(frozendict(configuration))

                budget[(frozendict(configuration), UpArrow)] += 1
            # Handle ↓
            elif set(second_configuration.keys()).issubset(inside_vertex - {vertex}):
                second_config_root = find_root(second_configuration, tree)
                # If raw, explicitly store raw second_configuration, otherwise store symbolic ↓ || child_id
                second_configuration = frozendict(second_configuration) if raw else DownArrow + str(second_config_root)
                valid_transitions[frozendict(configuration)].add(second_configuration)
                # Add the other direction as second_configuration won't be enumerated
                valid_transitions[second_configuration].add(frozendict(configuration))

                budget[(frozendict(configuration), DownArrow + str(second_config_root))] += 1
            else:
                valid_transitions[frozendict(configuration)].add(frozendict(second_configuration))
                # The other direction will also be covered when reaching configuration=second_configuration
        if raw:
            # Account for the that it's possible to get back from a different ↓ || child_id configuration,
            # potentially to a different configuration.
            down_configs_dict = defaultdict(lambda: set())
            down_configs = [second_configuration for second_configuration in enumerate_transitions(configuration, tree)
                            if set(second_configuration.keys()).issubset(inside_vertex - {vertex})]
            for down_config in down_configs:
                down_configs_dict[find_root(down_config, tree)].add(frozendict(down_config))
            for down_config in down_configs:
                for config in down_configs_dict[find_root(down_config, tree)]:
                    valid_transitions[frozendict(down_config)].update(valid_transitions[config])

    return valid_transitions, budget


def enumerate_signatures(vertex: str, tree: Tree, num_robots: int, raw: bool) -> List[Signature]:
    #  Let G=(V,E) be the following graph:
    #  1. V = {Configurations that occupy vertex} + {'↑'} + {Configurations projected to '↓'} (raw=True)
    #                                                     + {'↓'} (raw=False)
    #  2. E = an edge exists iff it is a valid transition
    # G is represented with an adjacency list valid_transitions:
    valid_transitions, budget = signatures_precompute(vertex, tree, num_robots, raw=raw)

    collected_signatures = []
    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    def update_used_transitions(current_signature: Signature, next_config: FormalConfiguration,
                                used_transitions: Dict[FormalConfiguration, Set[FormalConfiguration]]):
        if type(current_signature[-1]) is not str and type(next_config) is not str:
            used_transitions[current_signature[-1]].add(next_config)
        transition = (current_signature[-1], next_config)
        consumed_budget = sum(config1 == transition[0] and config2 == transition[1] for config1, config2 in zip(current_signature, current_signature[1:]))
        if consumed_budget + 1 == budget[transition] + budget[(transition[1], transition[0])]:
            used_transitions[current_signature[-1]].add(next_config)
        return used_transitions

    # We need to enumerate all possible paths in G that don't repeat an edge.
    # This corresponds to signatures where a transition does not repeat.
    def dfs_scan_signatures(current_signature: List[FormalConfiguration],
                            used_transitions: Dict[FormalConfiguration, Set[FormalConfiguration]]):
        if len(tree.children(vertex)) == 0 and sum(type(config) is not str for config in current_signature) > 1:
            # If vertex is a leaf, we can assume w.l.o.g that it is visited precisely once.
            # Indeed, connected configurations are collapsible.
            return
        if any(type(config) is not str for config in current_signature):
            # Add signature only if it visits vertex
            collected_signatures.append(current_signature)
        for next_config in valid_transitions[current_signature[-1]] - used_transitions[current_signature[-1]]:
            used_transitions = update_used_transitions(tuple(current_signature), next_config, used_transitions)
            next_signature = current_signature+[next_config]
            dfs_scan_signatures(next_signature, deepcopy(used_transitions))

    dfs_scan_signatures([start_config], used_transitions=defaultdict(lambda: set()))
    return collected_signatures

