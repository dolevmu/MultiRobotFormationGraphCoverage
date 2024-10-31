# from memory_profiler import profile
import gc
import hashlib
import cProfile
import pstats
import io

from collections import defaultdict, Counter
from frozendict import frozendict
from tqdm import tqdm

from treelib import Tree
from typing import Dict, NamedTuple, Optional

from trees.configuration import find_root, UpArrow, DownArrow
from trees.signature import Signature, project, enumerate_signatures, freeze_signature, get_child_key, _project, \
    unpack_signature, pack_signature
from trees.traversal import Traversal, is_traversal

PROFILING = False
# Global dictionary to store tables
tables = {}


class TableEntry(NamedTuple):
    vertex: str
    signature: Signature
    child_signatures: Dict[str, Signature]
    cost: int


Table = Dict[Signature, TableEntry]


def get_down_capacity(table: Table) -> int:
    max_capacity = 0
    for entry in table.values():
        signature = unpack_signature(entry.signature)
        capacity = signature.count(UpArrow) - (signature[-1] == UpArrow)
        if capacity > max_capacity:
            max_capacity = capacity
    return max_capacity


# @profile
def compute_table(vertex: str,
                  tree: Tree,
                  num_robots: int,
                  backtrack: bool = False,
                  heuristics_on: bool = True,
                  parallel: bool = True) -> Table:
    if not backtrack and PROFILING:
        pr = cProfile.Profile()
        # Start profiling
        pr.enable()

    parent = vertex if vertex == tree.root else tree.parent(vertex).identifier
    table = defaultdict(lambda: TableEntry(vertex=vertex, signature=(), child_signatures={}, cost=2*tree.size()))

    # Recursively Compute tables of children.
    children_tables = {child.identifier: compute_table(child.identifier, tree, num_robots, backtrack=backtrack,
                                                       heuristics_on=heuristics_on) for child in tree.children(vertex)}
    down_capacities = {DownArrow + child: get_down_capacity(table) for child, table in children_tables.items()}
    # Enumerate signatures at vertex:
    signatures_iterator = enumerate_signatures(vertex, tree, num_robots, raw=False,
                                               global_arrow_capacities=down_capacities,
                                               heuristics_on=heuristics_on, parallel=parallel)

    for packed_signature in tqdm(signatures_iterator, desc=f"Vertex={vertex: >4}"):
        signature = unpack_signature(packed_signature)

        matched_keys = True
        cost = sum(find_root(config, tree) == vertex for config in signature if type(config) is not str)
        child_signatures = {}

        for child in tree.children(vertex):
            child_key = hashlib.sha256(pack_signature(freeze_signature(get_child_key(vertex, child.identifier, signature, tree)))).hexdigest()
            if not child_key in children_tables[child.identifier]:
                matched_keys = False
                break
            cost += children_tables[child.identifier][child_key].cost
            if backtrack:
                child_signatures[child.identifier] = children_tables[child.identifier][child_key]
        if not matched_keys:
            # must find the projection of signature to all children
            continue

        # Add signature to the table
        signature_key = hashlib.sha256(pack_signature(freeze_signature(project(parent, signature, tree)))).hexdigest()
        if cost < table[signature_key].cost:
            # If found a signature with a smaller key, update table
            # TODO: we may have an even better condition: just consider configs before and after '↑' as the key
            table[signature_key] = TableEntry(vertex, packed_signature, child_signatures, cost)

    # Free memory...
    for child_table in children_tables.values():
        del child_table
        gc.collect()
    del down_capacities
    gc.collect()

    if not backtrack and PROFILING:
        # Stop profiling
        pr.disable()

        # Print profiling stats to a string
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time

        # Output the profiling results to console
        print(f"Profiling results for vertex {vertex}:\n")
        print(s.getvalue())

    return table


# @profile
def compute_table_root_dfs(tree: Tree, num_robots: int, backtrack: bool = False) -> Table:
    if PROFILING:
        pr = cProfile.Profile()
        pr.enable()

    # Initialize stack with the root node and a flag indicating it’s not ready for processing
    stack = [(tree.root, False)]
    visited = set()

    while stack:
        vertex, ready = stack.pop()

        # If ready to process (all children have been processed)
        if ready:
            parent = vertex if vertex == tree.root else tree.parent(vertex).identifier
            table = defaultdict(
                lambda: TableEntry(vertex=vertex, signature=(), child_signatures={}, cost=2 * tree.size()))
            children_tables = {child.identifier: tables[child.identifier] for child in tree.children(vertex) if
                               child.identifier in tables}
            down_capacities = {DownArrow + child: get_down_capacity(table) for child, table in children_tables.items()}
            signatures_iterator = enumerate_signatures(vertex, tree, num_robots, raw=False,
                                                       global_arrow_capacities=down_capacities)

            for packed_signature in tqdm(signatures_iterator, desc=f"Vertex={vertex: >4}"):
                signature = unpack_signature(packed_signature)
                matched_keys = True
                cost = sum(find_root(config, tree) == vertex for config in signature if type(config) is not str)
                child_signatures = {}

                for child in tree.children(vertex):
                    child_key = hashlib.sha256(pack_signature(
                        freeze_signature(get_child_key(vertex, child.identifier, signature, tree)))).hexdigest()
                    if child_key not in children_tables[child.identifier]:
                        matched_keys = False
                        break
                    cost += children_tables[child.identifier][child_key].cost
                    if backtrack:
                        child_signatures[child.identifier] = children_tables[child.identifier][child_key]

                if not matched_keys:
                    continue

                signature_key = hashlib.sha256(
                    pack_signature(freeze_signature(project(parent, signature, tree)))).hexdigest()
                if cost < table[signature_key].cost:
                    table[signature_key] = TableEntry(vertex, packed_signature, child_signatures, cost)

            # Store the completed table for this node
            tables[vertex] = table

            # Free memory for children tables
            for child in tree.children(vertex):
                if child.identifier in tables:
                    del tables[child.identifier]
            gc.collect()
            del down_capacities
            gc.collect()

        # If not ready, push it back onto the stack as ready, then push all children
        else:
            # Mark as ready for processing
            stack.append((vertex, True))
            for child in tree.children(vertex):
                if child.identifier not in visited:
                    stack.append((child.identifier, False))
                    visited.add(child.identifier)

    if PROFILING:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())

    # Return the root's table as the final result
    return tables[tree.root]


def _reconstruct(table_entry: TableEntry, tree: Tree) -> Traversal:
    # Stop condition: for leaves the partial solution is the signature
    if len(tree.children(table_entry.vertex)) == 0:
        return table_entry.signature
    # We must follow the pointers to complete table_entry.signature into a partial solution
    partial_solutions = {child: _reconstruct(child_entry, tree) for child, child_entry in table_entry.child_signatures.items()}

    # Now we must compose the partial solution, by completing each down arrow with the corresponding sequence
    # from the child partial solution.
    partial_solution = []
    for formal_config in unpack_signature(table_entry.signature):
        if type(formal_config) is not str:
            partial_solution.append(formal_config)
        elif formal_config == UpArrow:
            partial_solution.append(formal_config)
        else:  # Here, we should take a sequence from a corresponding child
            child = formal_config[1:]
            if type(partial_solutions[child]) is bytes:
                child_partial_solution = unpack_signature(partial_solutions[child])
            else:
                child_partial_solution = partial_solutions[child]
            parent_project = _project(table_entry.vertex, child_partial_solution, tree)
            parent_project = list(parent_project) + [UpArrow]
            start = parent_project.index(DownArrow + child)
            end = min(idx for idx, formal_config in enumerate(parent_project[start+1:], start+1)
                      if formal_config == UpArrow or
                      (type(formal_config) is not str and formal_config[table_entry.vertex] > 0))
            partial_solution = partial_solution + list(child_partial_solution)[start:end]
            if end + 1 < len(child_partial_solution):
                partial_solutions[child] = pack_signature(child_partial_solution[end+1:])
    return partial_solution


def reconstruct(table_entry: TableEntry, tree: Tree) -> Traversal:
    traversal = _reconstruct(table_entry, tree)
    return tuple(Counter(config) for config in traversal)


def fpt_compute_traversal_time(tree: Tree, num_robots: int) -> Traversal:
    table = compute_table(tree.root, tree, num_robots, backtrack=False)

    # Find traversal with minimal cost
    traversal_time = 2*tree.size()
    for table_entry in table.values():
        if table_entry.cost < traversal_time:
            traversal_time = table_entry.cost
    return traversal_time

def compute_single_robot_traversal_time(tree: Tree):
    def dfs_visit(vertex: str):
        children = tree.children(vertex)
        if len(children) > 1:
            return max(dfs_visit(child.identifier) for child in children)
        elif len(children) == 1:
            return dfs_visit(children[0].identifier) + 1
        else:
            return 1


def compute_single_robot_traversal(tree: Tree, backtrack: bool = True):
    def _compute_single_robot_traversal(tree: Tree):
        traversal = [Counter({tree.root: 1})]
        for _, child in sorted([(tree.subtree(child.identifier).depth(), child.identifier)
                                for child in tree.children(tree.root)]):
            sub_traversal = _compute_single_robot_traversal(tree.subtree(child))
            traversal = traversal + sub_traversal
        return traversal

    if not backtrack:
        return 2 * tree.size() - tree.depth()

    traversal = _compute_single_robot_traversal(tree)
    return traversal[:2 * tree.size() - tree.depth()]


def fpt_compute_traversal(tree: Tree,
                          num_robots: int,
                          backtrack: bool = True,
                          heuristics_on: bool = True,
                          parallel: bool = True) -> Optional[Traversal]:
    table = compute_table(tree.root, tree, num_robots,
                          backtrack=backtrack, heuristics_on=heuristics_on, parallel=parallel)
    # Find traversal with minimal cost
    root_table_entry = None
    traversal_time = 2*tree.size()
    for table_entry in table.values():
        if table_entry.cost < traversal_time:
            traversal_time = table_entry.cost
            root_table_entry = table_entry

    if root_table_entry is None:
        return None

    if backtrack:
        # Reconstruct traversal from root_signature
        traversal = reconstruct(root_table_entry, tree)
        assert is_traversal(traversal, tree)
        assert all(sum(config.values()) == num_robots for config in traversal)
        assert len(traversal) == root_table_entry.cost
        return traversal
    else:
        return traversal_time
