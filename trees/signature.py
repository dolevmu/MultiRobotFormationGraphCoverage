import hashlib

from memory_profiler import profile
from collections import Counter, defaultdict
from functools import reduce

import msgpack

from frozendict import frozendict
from treelib import Tree
from typing import Tuple, Dict, Set, List, Optional, Iterator, NamedTuple

from trees.configuration import is_connected, enumerate_configurations, find_root, UpArrow, DownArrow, \
    FormalConfiguration, FrozenFormalConfiguration, FormalTransition, pack_configuration, unpack_configuration
from trees.transition import is_transition, enumerate_transitions, is_up_transition

Signature = Tuple[FormalConfiguration, ...]
FrozenSignature = Tuple[FrozenFormalConfiguration, ...]

class TableEntry(NamedTuple):
    vertex: str
    signature: Signature
    child_signatures: Dict[str, Signature]
    cost: int


Table = Dict[Signature, TableEntry]

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
        elif set(configuration.keys()).issubset(set(tree.expand_tree(vertex))):
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
    successors = set(tree.expand_tree(vertex))
    ancestors = {ancestor for ancestor in tree.nodes if ancestor not in set(tree.expand_tree(vertex))} | {vertex}
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
    inside_vertex = set(tree.expand_tree(vertex))
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

# @profile # won't work for iterator, need to wrap with a function
def enumerate_signatures(vertex: str,
                         tree: Tree,
                         num_robots: int,
                         raw: bool = False,
                         global_arrow_capacities: Optional[Dict[str, int]] = None,
                         max_sig_length: Optional[int] = None,
                         heuristics_on: bool = True) -> Iterator[Signature]:
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
            global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow]) + 1

    if tree.parent(vertex) and UpArrow not in global_arrow_capacities:
        global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow]) + 1

    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    if max_sig_length is None:
        max_sig_length = tree.size()  # always holds

    if heuristics_on:
        max_sig_length = min(max_sig_length, 10)

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
    def dfs_scan_signatures(current_signature: List[FormalConfiguration], max_sig_length: int):
        if len(tree.children(vertex)) == 0 and sum(type(config) is not str for config in current_signature) > 1:
            # If vertex is a leaf, we can assume w.l.o.g that it is visited precisely once.
            # Indeed, connected configurations are collapsible.
            return
        if max_sig_length < len([config for config in current_signature if type(config) is not str]):
            return

        if any(type(config) is not str for config in current_signature):
            # Add signature only if it visits vertex
            yield pack_signature(tuple(current_signature))

            # Heuristic: only get out once
            if current_signature.count(UpArrow) == 2 and heuristics_on:
                return

        used_transitions = compute_used_transitions(current_signature)
        next_configs = valid_transitions[current_signature[-1]] - used_transitions

        covered = set(reduce(set.union, [config.keys() for config in current_signature if type(config) is not str], set()))
        if all(
                DownArrow + child.identifier in current_signature
                or set(tree.expand_tree(child.identifier)).issubset(covered)
                for child in tree.children(vertex)):
            # WLOG, due to collapsability, if searched all children, only go up
            next_configs = [config for config in next_configs
                            if is_up_transition(vertex, (current_signature[-1], config), tree, valid_transitions[current_signature[-1]])]
        elif heuristics_on and num_robots > 1:
            # Find all children that are partly covered
            visited = {child.identifier for child in tree.children(vertex)
                       if DownArrow + child.identifier not in current_signature and
                       not set(tree.expand_tree(child.identifier)).issubset(covered)
                       and child.identifier in covered}

            if visited:  # If decided to search two children in parallel, must follow this route
                # This is not a heuristic, due to collapsability
                next_real_configs = [config for config in next_configs
                                     if type(config) is not str and visited.issubset(config)]
                next_symbolic_configs = []  # Due to connectivity must be empty
                if not next_real_configs:  # True WLOG. If can't go further in parallel, could have avoided it altogether.
                    return

            # Heuristic: if didn't search all children, must cover a new node *when possible*
            else:
                covered = covered | {config for config in current_signature if type(config) is str} - {UpArrow}
                to_cover = set(tree.expand_tree(vertex)) - covered
                next_real_configs = [config for config in next_configs
                                     if type(config) is not str and (set(config.keys()) & to_cover)]
                next_symbolic_configs = [config for config in next_configs if type(config) is str
                                         and not config in covered and config != UpArrow]

            if len(next_real_configs) + len(next_symbolic_configs) > 0:
                next_configs = next_real_configs + next_symbolic_configs

            elif num_robots > 2:
                return # We can always avoid that. If we don't we will get a lot of garbage sigs.

            # Heuristic: if robots can go down, and there is enough ground to cover, do it
            # if next_symbolic_configs:
            #     big_children = list(sorted(child for child in next_symbolic_configs
            #                                if len(set(tree.expand_tree(child[1:]))) >= num_robots))
            #     if big_children:
            #         # Explore a child
            #         big_child = big_children[0]
            #         next_configs = [big_child] + [config for config in next_real_configs if big_child[1:] in config]

        for next_config in next_configs:
            next_signature = current_signature[:]+[next_config]
            yield from dfs_scan_signatures(next_signature, max_sig_length)

    yield from dfs_scan_signatures([start_config],
                                   max_sig_length=max_sig_length)


def enumerate_signatures_given_child_tables(children_tables: Dict[str, Table],
                                            vertex: str,
                                            tree: Tree,
                                            num_robots: int,
                                            raw: bool,
                                            global_arrow_capacities: Optional[Dict[str, int]] = None,
                                            max_sig_length: Optional[int] = None,
                                            heuristics_on: bool = True) -> Iterator[Signature]:
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
            global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow]) + 1

    if tree.parent(vertex) and UpArrow not in global_arrow_capacities:
        global_arrow_capacities[UpArrow] = len(valid_transitions[UpArrow]) + 1

    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    if max_sig_length is None:
        max_sig_length = tree.size()  # always holds

    if heuristics_on:
        max_sig_length = min(max_sig_length, 8)

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
    def dfs_scan_signatures(current_signature: List[FormalConfiguration], max_sig_length: int):
        if len(tree.children(vertex)) == 0 and sum(type(config) is not str for config in current_signature) > 1:
            # If vertex is a leaf, we can assume w.l.o.g that it is visited precisely once.
            # Indeed, connected configurations are collapsible.
            return
        if max_sig_length < len([config for config in current_signature if type(config) is not str]):
            # No need to look at signatures longer than the upper bound
            # A tight bound for k=2 is tree.size(): consider a star graph
            return

        if any(type(config) is not str for config in current_signature):
            # Add signature only if has a real config.
            # Also check it matches all child tables if there are any.

            matched_keys = True
            for child in tree.children(vertex):
                child_key = pack_signature(freeze_signature(get_child_key(vertex, child.identifier, tuple(current_signature), tree)))
                if child_key not in children_tables[child.identifier]:
                    matched_keys = False
                    break
            if matched_keys:
                yield pack_signature(tuple(current_signature))

        # Heuristic: only get out once
        if current_signature.count(UpArrow) == 2 and heuristics_on:
            return

        used_transitions = compute_used_transitions(current_signature)
        next_configs = valid_transitions[current_signature[-1]] - used_transitions

        covered = set(reduce(set.union, [config.keys() for config in current_signature if type(config) is not str], set()))
        if all(
                DownArrow + child.identifier in current_signature
                or set(tree.expand_tree(child.identifier)).issubset(covered)
                for child in tree.children(vertex)):
            # WLOG, due to collapsability, if searched all children, only go up
            next_configs = [config for config in next_configs
                            if is_up_transition(vertex, (current_signature[-1], config), tree, valid_transitions[current_signature[-1]])]
        elif heuristics_on and num_robots > 1:
            # Heuristic: if didn't search all children, must cover a new node *when possible*
            covered = covered | {config for config in current_signature if type(config) is str} - {UpArrow}
            to_cover = set(tree.expand_tree(vertex)) - covered
            next_real_configs = [config for config in next_configs
                                 if type(config) is not str and (set(config.keys()) & to_cover)]

            next_symbolic_configs = [config for config in next_configs if type(config) is str
                                     and not config in covered and config != UpArrow]

            if len(next_real_configs) + len(next_symbolic_configs) > 0:
                next_configs = next_real_configs + next_symbolic_configs
            elif num_robots > 2:
                return # We can always avoid that. If we don't we will get a lot of garbage sigs.

            # Heuristic: if robots can go down, and there is enough ground to cover, do it
            if next_symbolic_configs:
                big_children = [child for child in next_symbolic_configs
                                if len(tree.expand_tree(child[1:])) >= num_robots]
                if big_children:
                    next_configs = big_children

        for next_config in next_configs:
            next_signature = current_signature[:]+[next_config]
            yield from dfs_scan_signatures(next_signature, max_sig_length)


    yield from dfs_scan_signatures([start_config], max_sig_length=max_sig_length)

def pack_signature(signature: Signature):
    return msgpack.packb(tuple(pack_configuration(config) for config in signature))


def unpack_signature(packed_signature) -> Signature:
    return tuple(unpack_configuration(packed_config) for packed_config in msgpack.unpackb(packed_signature))