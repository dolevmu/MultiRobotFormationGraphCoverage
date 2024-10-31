from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import reduce

import msgpack
import numpy as np

from frozendict import frozendict
from treelib import Tree
from typing import Tuple, Dict, Set, List, Optional, Iterator

from trees.configuration import enumerate_configurations, find_root, UpArrow, DownArrow, \
    FormalConfiguration, FormalTransition
from trees.transition import enumerate_transitions, is_up_transition

from trees.signature import Signature, pack_signature


def signatures_precompute(vertex: str, tree: Tree, num_robots: int, raw: bool
                          ) -> Tuple[Dict[FormalConfiguration, Set[FormalConfiguration]], Dict[FormalTransition, int]]:
    inside_vertex = tree.subtree(vertex).nodes.keys()
    outside_vertex = tree.nodes.keys() - inside_vertex

    # Pre-computation: get all configurations that occupy vertex
    valid_configurations = enumerate_configurations(vertex, tree, num_robots)

    # Pre-computation: for each configuration, get list of valid transitions
    valid_transitions = dict()

    # Pre-computation: for each configuration, get the budget for each arrow: how many such transitions are available
    budget = Counter()

    for configuration in valid_configurations:
        collected_transitions = enumerate_transitions(configuration, tree)

        for second_configuration in collected_transitions:
            # Handle ↑
            if set(second_configuration.keys()).issubset(outside_vertex):
                valid_transitions[frozendict(configuration)] = valid_transitions.get(frozendict(configuration), set()) | {UpArrow}
                valid_transitions[UpArrow] = valid_transitions.get(UpArrow, set()) | {frozendict(configuration)}

                budget[(frozendict(configuration), UpArrow)] += 1
            # Handle ↓
            elif set(second_configuration.keys()).issubset(inside_vertex - {vertex}):
                second_config_root = find_root(second_configuration, tree)
                # If raw, explicitly store raw second_configuration, otherwise store symbolic ↓ || child_id
                second_configuration = frozendict(second_configuration) if raw else DownArrow + str(second_config_root)
                valid_transitions[frozendict(configuration)] = valid_transitions.get(frozendict(configuration), set()) | {second_configuration}
                # Add the other direction as second_configuration won't be enumerated
                valid_transitions[second_configuration] = valid_transitions.get(second_configuration, set()) | {frozendict(configuration)}

                budget[(frozendict(configuration), DownArrow + str(second_config_root))] += 1
            else:
                valid_transitions[frozendict(configuration)] = valid_transitions.get(frozendict(configuration), set()) | {frozendict(second_configuration)}
                # The other direction will also be covered when reaching configuration=second_configuration
        if raw:
            # Account for the that it's possible to get back from a different ↓ || child_id configuration,
            # potentially to a different configuration.
            down_configs_dict = dict()
            down_configs = [second_configuration for second_configuration in enumerate_transitions(configuration, tree)
                            if set(second_configuration.keys()).issubset(inside_vertex - {vertex})]
            for down_config in down_configs:
                down_configs_dict[find_root(down_config, tree)] = down_configs_dict.get(find_root(down_config, tree), set()) | {frozendict(down_config)}
            for down_config in down_configs:
                for config in down_configs_dict[find_root(down_config, tree)]:
                    valid_transitions[frozendict(down_config)] = valid_transitions.get(frozendict(down_config), set()) | valid_transitions.get(config, set())

    return valid_transitions, budget


def compute_used_transitions(current_signature: List[FormalConfiguration],
                             budget: Dict[FormalTransition, int],
                             global_arrow_capacities: Optional[Dict[str, int]] = None) -> Set[FormalConfiguration]:
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

def parallel_dfs_scan_signatures(vertex: str,
                                 tree: Tree,
                                 num_robots: int,
                                 current_signature: List[FormalConfiguration],
                                 next_configs: List[FormalConfiguration],
                                 valid_transitions: Dict[FormalConfiguration, Set[FormalConfiguration]],
                                 budget: Dict[FormalTransition, int],
                                 global_arrow_capacities: Optional[Dict[str, int]] = None,
                                 max_sig_length: Optional[int] = None,
                                 heuristics_on: bool = True):
    with ProcessPoolExecutor() as executor:
        # Submit each configuration to the executor
        futures = [
            executor.submit(worker_dfs, vertex, tree, num_robots, current_signature, next_config, valid_transitions, budget, global_arrow_capacities, max_sig_length, heuristics_on)
            for next_config in next_configs
        ]

        # Collect and yield results as they complete
        for future in futures:
            results = future.result()
            for result in results:
                yield result


# We need to enumerate all possible paths in G that don't repeat an edge.
# This corresponds to signatures where a transition does not repeat.
def dfs_scan_signatures(vertex: str,
                        tree: Tree,
                        num_robots: int,
                        current_signature: List[FormalConfiguration],
                        valid_transitions: Dict[FormalConfiguration, Set[FormalConfiguration]],
                        budget: Dict[FormalTransition, int],
                        global_arrow_capacities: Optional[Dict[str, int]] = None,
                        max_sig_length: Optional[int] = None,
                        heuristics_on: bool = True):
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
        if current_signature.count(UpArrow) == 2 and heuristics_on:
            return

    used_transitions = compute_used_transitions(current_signature, budget, global_arrow_capacities)
    next_configs = valid_transitions.get(current_signature[-1], set()) - used_transitions

    covered = set(reduce(set.union, [config.keys() for config in current_signature if type(config) is not str], set()))
    if all(
            DownArrow + child.identifier in current_signature
            or set(tree.subtree(child.identifier).nodes.keys()).issubset(covered)
            for child in tree.children(vertex)):
        # WLOG, due to collapsability, if searched all children, only go up
        next_configs = [config for config in next_configs
                        if is_up_transition(vertex, (current_signature[-1], config), tree, valid_transitions.get(current_signature[-1], set()))]
    elif heuristics_on and num_robots > 1:
        # Heuristic: if didn't search all children, must cover a new node *when possible*
        covered = covered | {config for config in current_signature if type(config) is str} - {UpArrow}
        next_real_configs = [config for config in next_configs
                             if type(config) is not str and not set(config.keys()).issubset(covered)]
        next_symbolic_configs = [config for config in next_configs if type(config) is str and not config in covered]
        if len(next_real_configs) + len(next_symbolic_configs) > 0:
            next_configs = next_real_configs + next_symbolic_configs

    yield from parallel_dfs_scan_signatures(vertex, tree, num_robots, current_signature, next_configs, valid_transitions, budget, global_arrow_capacities, max_sig_length, heuristics_on)



def worker_dfs(vertex: str,
               tree: Tree,
               num_robots: int,
               current_signature: List[FormalConfiguration],
               next_config: FormalConfiguration,
               valid_transitions: Dict[FormalConfiguration, Set[FormalConfiguration]],
               budget: Dict[FormalTransition, int],
               global_arrow_capacities: Optional[Dict[str, int]] = None,
               max_sig_length: Optional[int] = None,
               heuristics_on: bool = True):
                # (next_config, current_signature, max_sig_length):
    next_signature = current_signature[:] + [next_config]
    # yield from dfs_scan_signatures(vertex, tree, num_robots, next_signature, valid_transitions, budget, global_arrow_capacities, max_sig_length, heuristics_on)
    return list(dfs_scan_signatures(vertex, tree, num_robots, next_signature, valid_transitions, budget, global_arrow_capacities, max_sig_length, heuristics_on))


def parallel_enumerate_signatures(vertex: str,
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
            global_arrow_capacities[DownArrow + child.identifier] = len(valid_transitions.get(DownArrow + child.identifier, set()))
        if tree.parent(vertex):
            global_arrow_capacities[UpArrow] = len(valid_transitions.get(UpArrow, set())) + 1

    if tree.parent(vertex) and UpArrow not in global_arrow_capacities:
        global_arrow_capacities[UpArrow] = len(valid_transitions.get(UpArrow, set())) + 1

    start_config = frozendict(Counter({vertex: num_robots})) if vertex == tree.root else UpArrow

    if max_sig_length is None:
        max_sig_length = tree.size()  # always holds

    if heuristics_on:  # or True:
        max_sig_length = min(max_sig_length, 8)

    yield from dfs_scan_signatures(vertex, tree, num_robots, [start_config],
                                   valid_transitions, budget, global_arrow_capacities=global_arrow_capacities, max_sig_length=max_sig_length, heuristics_on=heuristics_on)