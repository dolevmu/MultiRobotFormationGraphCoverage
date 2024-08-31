from collections import defaultdict, Counter

from frozendict import frozendict
from treelib import Tree
from typing import Dict, NamedTuple

from trees.configuration import enumerate_configurations, find_root
from trees.signature import Signature, project, enumerate_signatures, UpArrow, freeze_signature, DownArrow
from trees.traversal import Traversal, is_traversal


class TableEntry(NamedTuple):
    signature: Signature
    child_signatures: Dict[str, Signature]
    cost: int


Table = Dict[Signature, TableEntry]


def compute_table(vertex: str, tree: Tree, num_robots: int) -> Table:
    parent = vertex if vertex == tree.root else tree.parent(vertex).identifier
    table = defaultdict(lambda: TableEntry(signature=(), child_signatures={}, cost=2*tree.size()))

    # Recursively Compute tables of children.
    children_tables = {child.identifier: compute_table(child.identifier, tree, num_robots) for child in tree.children(vertex)}

    # Enumerate signatures at vertex:
    for signature in enumerate_signatures(vertex, tree, num_robots, raw=False):
        if signature == [frozendict({'': 2}),
                         DownArrow + '0',
                         frozendict({'': 2}),
                         DownArrow + '1']:
            print(signature)

        matched_keys = True
        cost = sum(find_root(config, tree) == vertex for config in signature if type(config) is not str)
        child_signatures = {}

        for child in tree.children(vertex):
            child_key = freeze_signature(project(child.identifier, signature, tree))
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
            table[signature_key] = TableEntry(signature, child_signatures, cost)
            # TODO: we may have an even better condition: just consider configs before and after '↑' as the key
    return table

def reconstruct(table_entry: TableEntry) -> Traversal:
    # We must follow the pointers to complete table_entry.signature into a partial solution
    partial_solutions = {child: reconstruct(child_entry) for child, child_entry in table_entry.child_signatures}
    root_signature = table_entry.signature
    return tuple(list(root_signature) + list(partial_solutions.values()))


def fpt_compute_traversal_time(tree: Tree, num_robots: int) -> Traversal:
    my_sig = (frozendict({"": 2}),
              DownArrow,
              frozendict({"": 2}),
              DownArrow)
    table = compute_table(tree.root, tree, num_robots)
    # Find traversal with minimal cost
    traversal_time = 2*tree.size()
    for table_entry in table.values():
        if table_entry.cost < traversal_time:
            traversal_time = table_entry.cost
    return traversal_time


def fpt_compute_traversal(tree: Tree, num_robots: int) -> Traversal:
    table = compute_table(tree.root, tree, num_robots)
    # Find traversal with minimal cost
    root_signature = None
    traversal_time = 2*tree.size()
    for table_entry in table.values():
        if table_entry.cost < traversal_time:
            traversal_time = table_entry.cost
            root_signature = table_entry.signature

    # Reconstruct traversal from root_signature
    traversal = reconstruct(root_signature)

    assert is_traversal(traversal, tree)
    assert all(config.total() == num_robots for config in traversal)
    assert len(traversal) == traversal_time
    return traversal
