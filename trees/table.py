from collections import defaultdict, Counter

from frozendict import frozendict
from tqdm import tqdm

from treelib import Tree
from typing import Dict, NamedTuple, Optional

from trees.configuration import find_root, UpArrow, DownArrow
from trees.signature import Signature, project, enumerate_signatures, freeze_signature, get_child_key, _project
from trees.traversal import Traversal, is_traversal


class TableEntry(NamedTuple):
    vertex: str
    signature: Signature
    child_signatures: Dict[str, Signature]
    cost: int


Table = Dict[Signature, TableEntry]


def get_down_capacity(table: Table) -> int:
    max_capacity = 0
    for entry in table.values():
        capacity = entry.signature.count(UpArrow) - (entry.signature[-1] == UpArrow)
        if capacity > max_capacity:
            max_capacity = capacity
    return max_capacity


def compute_table(vertex: str, tree: Tree, num_robots: int) -> Table:
    parent = vertex if vertex == tree.root else tree.parent(vertex).identifier
    table = defaultdict(lambda: TableEntry(vertex=vertex, signature=(), child_signatures={}, cost=2*tree.size()))

    # Recursively Compute tables of children.
    children_tables = {child.identifier: compute_table(child.identifier, tree, num_robots) for child in tree.children(vertex)}
    down_capacities = {child: get_down_capacity(table) for child, table in children_tables.items()}
    # Enumerate signatures at vertex:
    signatures_at_vertex = enumerate_signatures(vertex, tree, num_robots, raw=False, down_capacities=down_capacities)
    num_signatures_at_vertex = len(signatures_at_vertex)
    for signature in tqdm(signatures_at_vertex, total=num_signatures_at_vertex, desc=f"Vertex={vertex: >4}"):
        if signature == [frozendict({'': 3}),
                         DownArrow + '0',
                         frozendict({'': 1, '0': 1, '01': 1}),
                         frozendict({'': 1, '0': 1, '2': 1}),
                         frozendict({'': 2, '1': 1})]:
            print('here')

        if signature == [UpArrow,
                         frozendict({'0': 3}),
                         frozendict({'0': 1, '00': 1, '01': 1}),
                         frozendict({'0': 1, '01': 1, '010': 1}),
                         DownArrow + '01',
                         frozendict({'0': 1, '01': 1, '011': 1}),
                         frozendict({'': 1, '0': 1, '01': 1}),
                         frozendict({'': 1, '0': 1, '2': 1}),
                         UpArrow]:
            print('here')

        if signature == [UpArrow,
                         frozendict({'0': 1, '00': 1, '01': 1}),
                         frozendict({'0': 1, '01': 1, '010': 1}),
                         frozendict({'01': 1, '010': 1, '0100': 1}),
                         frozendict({'01': 1, '010': 1, '011': 1}),
                         frozendict({'01': 1, '011': 1, '0110': 1}),
                         frozendict({'01': 1, '011': 1, '0111': 1}),
                         frozendict({'01': 1, '011': 1, '0112': 1}),
                         frozendict({'0': 1, '01': 1, '011': 1}),
                         frozendict({'': 1, '0': 1, '01': 1}),
                         UpArrow]:
            print('here')

        matched_keys = True
        cost = sum(find_root(config, tree) == vertex for config in signature if type(config) is not str)
        child_signatures = {}

        for child in tree.children(vertex):
            child_key = freeze_signature(get_child_key(vertex, child.identifier, signature, tree))
            if not child_key in children_tables[child.identifier]:
                matched_keys = False
                break
            cost += children_tables[child.identifier][child_key].cost
            child_signatures[child.identifier] = children_tables[child.identifier][child_key]
        if not matched_keys:
            # must find the projection of signature to all children
            continue

        # Add signature to the table
        signature_key = freeze_signature(project(parent, signature, tree))
        if cost < table[signature_key].cost:
            # If found a signature with a smaller key, update table
            table[signature_key] = TableEntry(vertex, signature, child_signatures, cost)
            # TODO: we may have an even better condition: just consider configs before and after '↑' as the key
    return table


def _reconstruct(table_entry: TableEntry, tree: Tree) -> Traversal:
    # Stop condition: for leaves the partial solution is the signature
    if len(tree.children(table_entry.vertex)) == 0:
        return table_entry.signature
    # We must follow the pointers to complete table_entry.signature into a partial solution
    partial_solutions = {child: _reconstruct(child_entry, tree) for child, child_entry in table_entry.child_signatures.items()}

    # Now we must compose the partial solution, by completing each down arrow with the corresponding sequence
    # from the child partial solution.
    partial_solution = []
    for formal_config in table_entry.signature:
        if type(formal_config) is not str:
            partial_solution.append(formal_config)
        elif formal_config == UpArrow:
            partial_solution.append(formal_config)
        else:  # Here, we should take a sequence from a corresponding child
            child = formal_config[1:]
            parent_project = _project(table_entry.vertex, partial_solutions[child], tree)
            parent_project = list(parent_project) + [UpArrow]
            start = parent_project.index(DownArrow + child)
            end = min(idx for idx, formal_config in enumerate(parent_project[start+1:], start+1)
                      if formal_config == UpArrow or
                      (type(formal_config) is not str and formal_config[table_entry.vertex] > 0))
            partial_solution = partial_solution + partial_solutions[child][start:end]
            if end + 1 < len(partial_solutions[child]):
                partial_solutions[child] = partial_solutions[child][end+1:]
    return partial_solution


def reconstruct(table_entry: TableEntry, tree: Tree) -> Traversal:
    traversal = _reconstruct(table_entry, tree)
    return tuple(Counter(config) for config in traversal)


def fpt_compute_traversal_time(tree: Tree, num_robots: int) -> Traversal:
    table = compute_table(tree.root, tree, num_robots)
    # Find traversal with minimal cost
    traversal_time = 2*tree.size()
    for table_entry in table.values():
        if table_entry.cost < traversal_time:
            traversal_time = table_entry.cost
    return traversal_time


def fpt_compute_traversal(tree: Tree, num_robots: int) -> Optional[Traversal]:
    table = compute_table(tree.root, tree, num_robots)
    # Find traversal with minimal cost
    root_table_entry = None
    traversal_time = 2*tree.size()
    for table_entry in table.values():
        if table_entry.cost < traversal_time:
            traversal_time = table_entry.cost
            root_table_entry = table_entry

    if root_table_entry is None:
        return None

    # Reconstruct traversal from root_signature
    traversal = reconstruct(root_table_entry, tree)

    assert is_traversal(traversal, tree)
    assert all(config.total() == num_robots for config in traversal)
    assert len(traversal) == root_table_entry.cost
    return traversal
