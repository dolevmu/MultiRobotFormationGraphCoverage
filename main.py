# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter

from trees.configuration import enumerate_configurations, is_connected
from trees.signature import project, is_signature, enumerate_signatures
from trees.table import compute_table, fpt_compute_traversal_time, fpt_compute_traversal
from trees.transition import enumerate_transitions
from trees.traversal import is_traversal
from trees.tree import example_tree, print_tree

from time import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tree = example_tree()
    print_tree(tree)

    # print('Configurations that occupy 1:')
    # for config in enumerate_configurations("1", tree, 3):
    #     print(config)
    #
    # print('Potential transitions from {"1": 1, "10": 1, "100": 1}:')
    # for config2 in enumerate_transitions(Counter({'0': 1, '01': 2}), tree):
    #     print(config2)
    #
    # traversal = (Counter({"": 1, "0": 1, "00": 1}),
    #              Counter({"": 1, "0": 1, "01": 1}),
    #              Counter({"": 1, "0": 1, "1": 1}),
    #              Counter({"": 1, "1": 1, "10": 1}),
    #              Counter({"1": 1, "10": 1, "100": 1}),
    #              Counter({"1": 1, "10": 1, "101": 1}),
    #              Counter({"10": 1, "101": 1, "1010": 1}))
    #
    # print(f"Traversal is {is_traversal(traversal, tree)}")
    #
    # verify_signatures = all(is_signature(project(vertex, traversal, tree), vertex, tree)
    #                         for vertex in tree.nodes)
    # print(f"Signatures are {verify_signatures}.")
    #
    # collected_signatures = enumerate_signatures("", tree, 2, raw=False)
    # print(len(collected_signatures))
    # print(collected_signatures[13])
    #
    # table = compute_table("0", tree, 2)
    # print(len(table))
    # for entry in table.values():
    #     print(entry.signature)
    start = time()
    traversal = fpt_compute_traversal(tree, 2)
    end = time()

    print(f"Traversal Time={len(traversal)}")
    print(f"Computation Time={end-start:.2f}")
    print(traversal)
