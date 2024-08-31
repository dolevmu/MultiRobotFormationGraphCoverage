# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter

from trees.configuration import enumerate_configurations, is_connected
from trees.signature import project, is_signature, enumerate_signatures
from trees.table import compute_table, fpt_compute_traversal_time
from trees.transition import enumerate_transitions
from trees.traversal import is_traversal
from trees.tree import example_tree, print_tree

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
    # collected_signatures = enumerate_signatures("0", tree, 2, raw=True)
    # print(len(collected_signatures))
    #
    # table = compute_table("0", tree, 2)
    # print('done')

    print(fpt_compute_traversal_time(tree, 2))
