from collections import Counter, defaultdict
from functools import reduce

import msgpack
import numpy as np

from frozendict import frozendict
from treelib import Tree
from typing import Tuple, Dict, Set, List, Optional, Iterator

from trees.configuration import is_connected, enumerate_configurations, find_root, UpArrow, DownArrow, \
    FormalConfiguration, FrozenFormalConfiguration, FormalTransition, pack_configuration, unpack_configuration
from trees.transition import is_transition, enumerate_transitions, is_up_transition

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
            sig.append(config)

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


def compute_signature_cost(vertex: str, signature: Signature, tree: Tree) -> int:
    return sum(find_root(config, tree) == vertex for config in signature if type(config) is not str)


def signatures_precompute(vertex: str, tree: Tree, num_robots: int, raw: bool
                          ) -> Tuple[Dict[FormalConfiguration, Set[FormalConfiguration]], Dict[FormalTransition, int]]:
    inside_vertex = tree.subtree(vertex).nodes.keys()
    outside_vertex = tree.nodes.keys() - inside_vertex

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


def enumerate_signatures(vertex: str,
                         tree: Tree,
                         num_robots: int,
                         raw: bool,
                         global_arrow_capacities: Optional[Dict[str, int]] = None,
                         max_sig_length: Optional[int] = None) -> Iterator[Signature]:
    # Let G=(V,E) be the following graph:
    # 1. V = {Configurations that occupy vertex} + {'↑'} + {Configurations projected to '↓'} (raw=True)
    #                                                    + {'↓'} (raw=False)
    # 2. E = an edge exists iff it is a valid transition
    # G is represented with an adjacency list valid_transitions:
    valid_transitions, budget = signatures_precompute(vertex, tree, num_robots, raw=raw)

    # down_capacity[vertex] specifies the maximal number of down arrows per child, overall.
    # E.g., due to collapsability, if child is a leaf in tree, down_capacity[child]=1, and there is only one transition
    # to this child.
    if global_arrow_capacities is None:
        global_arrow_capacities = dict()
        for child in tree.children(vertex):
            global_arrow_capacities[DownArrow + child.identifier] = len(valid_transitions[DownArrow + child.identifier])
        if tree.parent(vertex):
            global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow])

    if tree.parent(vertex) and UpArrow not in global_arrow_capacities:
        global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow])

    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    if max_sig_length is None:
        max_sig_length = tree.size()  # always holds

    max_sig_length = 8

    def compute_used_transitions(current_signature: List[FormalConfiguration]) -> Set[FormalConfiguration]:
        used_transitions = set()
        consumed_global_budget = Counter()
        for config in current_signature:
            if type(config) is str:
                consumed_global_budget[config] += 1
        for config, count in consumed_global_budget.items():
            if count == global_arrow_capacities[config]:
                used_transitions.add(config)

        consumed_budget = Counter()
        for i in range(1, len(current_signature)):
            if current_signature[i-1] == current_signature[-1]:
                if type(current_signature[i]) is str:
                    consumed_budget[current_signature[i]] += 1
                elif type(current_signature[-1]) is not str:
                    used_transitions.add(current_signature[i])
        for config, count in consumed_budget.items():
            if count == budget[(current_signature[-1], config)]:
                used_transitions.add(config)

        return used_transitions

    # We need to enumerate all possible paths in G that don't repeat an edge.
    # This corresponds to signatures where a transition does not repeat.
    def dfs_scan_signatures(current_signature: List[FormalConfiguration],
                            max_sig_length: int):
        if len(tree.children(vertex)) == 0 and sum(type(config) is not str for config in current_signature) > 1:
            # If vertex is a leaf, we can assume w.l.o.g that it is visited precisely once.
            # Indeed, connected configurations are collapsible.
            return
        if max_sig_length < len([config for config in current_signature if type(config) is not str]):
            # No need to look at signatures longer than the upper bound
            # A tight bound for k=2 is tree.size(): consider a star graph
            return

        if any(type(config) is not str for config in current_signature):
            # Add signature only if it visits vertex
            yield pack_signature(tuple(current_signature))

            # Heuristic: only get out once
            if current_signature.count(UpArrow) == 2:
                return

        used_transitions = compute_used_transitions(current_signature)
        next_configs = valid_transitions[current_signature[-1]] - used_transitions

        covered = set(reduce(set.union, [config.keys() for config in current_signature if type(config) is not str], set()))
        if all(
                DownArrow + child.identifier in current_signature
                or set(tree.subtree(child.identifier).nodes.keys()).issubset(covered)
                for child in tree.children(vertex)):
            # WLOG, due to collapsability, if searched all children, only go up
            next_configs = [config for config in next_configs
                            if is_up_transition(vertex, (current_signature[-1], config), tree, valid_transitions[current_signature[-1]])]
        else:
            # Heuristic: if didn't search all children, must cover a new node
            covered = covered | {config for config in current_signature if type(config) is str} - {UpArrow}
            next_real_configs = [config for config in next_configs
                                 if type(config) is not str and not set(config.keys()).issubset(covered)]
            next_configs = [config for config in next_configs
                            if type(config) is str and not config in covered] + next_real_configs

        for next_config in next_configs:
            next_signature = current_signature[:]+[next_config]
            yield from dfs_scan_signatures(next_signature, max_sig_length)

    yield from dfs_scan_signatures([start_config],
                                   max_sig_length=max_sig_length)


def enumerate_signatures_given_key(signature_key: Signature,
                                   vertex: str,
                                   tree: Tree,
                                   num_robots: int,
                                   raw: bool,
                                   global_arrow_capacities: Optional[Dict[str, int]] = None,
                                   max_sig_length: Optional[int] = None) -> Iterator[Signature]:

    valid_transitions, budget = signatures_precompute(vertex, tree, num_robots, raw=raw)

    if global_arrow_capacities is None:
        global_arrow_capacities = dict()
        for child in tree.children(vertex):
            global_arrow_capacities[DownArrow + child.identifier] = len(valid_transitions[DownArrow + child.identifier])
        if tree.parent(vertex):
            global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow])
    if tree.parent(vertex) and UpArrow not in global_arrow_capacities:
        global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow])

    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    if max_sig_length is None:
        max_sig_length = tree.size()  # always holds
    max_sig_length = 8

    def compute_used_transitions(current_signature: List[FormalConfiguration]) -> Set[FormalConfiguration]:
        used_transitions = set()
        consumed_global_budget = Counter()
        for config in current_signature:
            if type(config) is str:
                consumed_global_budget[config] += 1
        for config, count in consumed_global_budget.items():
            if count == global_arrow_capacities[config]:
                used_transitions.add(config)

        consumed_budget = Counter()
        for i in range(1, len(current_signature)):
            if current_signature[i-1] == current_signature[-1]:
                if type(current_signature[i]) is str:
                    consumed_budget[current_signature[i]] += 1
                elif type(current_signature[-1]) is not str:
                    used_transitions.add(current_signature[i])
        for config, count in consumed_budget.items():
            if count == budget[(current_signature[-1], config)]:
                used_transitions.add(config)

        return used_transitions

    # We need to enumerate all possible paths in G that don't repeat an edge.
    # This corresponds to signatures where a transition does not repeat.
    def dfs_scan_signatures(signature_key: Signature,
                            current_signature: List[FormalConfiguration],
                            max_sig_length: int):

        if len(tree.children(vertex)) == 0 and sum(type(config) is not str for config in current_signature) > 1:
            # If vertex is a leaf, we can assume w.l.o.g that it is visited precisely once.
            # Indeed, connected configurations are collapsible.
            return
        if max_sig_length < len([config for config in current_signature if type(config) is not str]):
            # No need to look at signatures longer than the upper bound
            # A tight bound for k=2 is tree.size(): consider a star graph
            return

        if any(type(config) is not str for config in current_signature):
            # Add signature only if it visits vertex
            parent = vertex if vertex == tree.root else tree.parent(vertex).identifier
            if project(parent, tuple(current_signature), tree) == signature_key:
                yield pack_signature(tuple(current_signature))

            # Heuristic: only get out once
            if current_signature.count(UpArrow) == 2:
                return

        used_transitions = compute_used_transitions(current_signature)
        next_configs = valid_transitions[current_signature[-1]] - used_transitions

        covered = set(reduce(set.union, [config.keys() for config in current_signature if type(config) is not str], set()))
        if all(
                DownArrow + child.identifier in current_signature
                or set(tree.subtree(child.identifier).nodes.keys()).issubset(covered)
                for child in tree.children(vertex)):
            # WLOG, due to collapsability, if searched all children, only go up
            next_configs = [config for config in next_configs
                            if is_up_transition(vertex, (current_signature[-1], config), tree, valid_transitions[current_signature[-1]])]
        else:
            # Heuristic: if didn't search all children, must cover a new node
            covered = covered | {config for config in current_signature if type(config) is str} - {UpArrow}
            next_real_configs = [config for config in next_configs
                                 if type(config) is not str and not set(config.keys()).issubset(covered)]
            next_configs = [config for config in next_configs
                            if type(config) is str and not config in covered] + next_real_configs

        for next_config in next_configs:
            # Here, we must verify we match the signature key. One way is to check the prefix matches:
            next_signature = current_signature[:]+[next_config]
            next_partial_key = project(vertex, tuple(next_signature), tree)
            if next_partial_key == tuple(list(signature_key)[:len(next_partial_key)]):
                yield from dfs_scan_signatures(signature_key, next_signature, max_sig_length)

    yield from dfs_scan_signatures(signature_key, [start_config], max_sig_length=max_sig_length)


def pack_signature(signature: Signature):
    return msgpack.packb(tuple(pack_configuration(config) for config in signature))


def unpack_signature(packed_signature) -> Signature:
    return tuple(unpack_configuration(packed_config) for packed_config in msgpack.unpackb(packed_signature))