# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter

from trees.signature import project, compute_signature_cost
from trees.table import fpt_compute_traversal
from trees.traversal import print_traversal, is_traversal
from trees.tree import hard_example_tree, example_tree, print_tree

from time import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tree = hard_example_tree()
    print_tree(tree)

    start = time()
    traversal = fpt_compute_traversal(tree, 3)
    end = time()

    print(f"Traversal Time={len(traversal)}, Tree Size={tree.size()}")
    print(f"Computation Time={end-start:.2f}")
    print_traversal(traversal)

    my_traver = (Counter({'': 3}),
                 Counter({'0': 3}),
                 Counter({'0': 1, '00': 1, '01': 1}),
                 Counter({'0': 1, '01': 1, '010': 1}),
                 Counter({'01': 1, '010': 1, '0100': 1}),
                 Counter({'01': 1, '010': 1, '011': 1}),
                 Counter({'01': 1, '011': 1, '0110': 1}),
                 Counter({'01': 1, '011': 1, '0111': 1}),
                 Counter({'01': 1, '011': 1, '0112': 1}),
                 Counter({'0': 1, '01': 1, '011': 1}),
                 Counter({'': 1, '0': 1, '01': 1}),
                 Counter({'': 1, '0': 1, '2': 1}),
                 Counter({'': 2, '1': 1}),
                 Counter({'1': 2, '10': 1}),
                 Counter({'10': 2, '100': 1}),
                 Counter({'10': 1, '101': 2}),
                 Counter({'101': 1, '1010': 1, '1011': 1}),
                 Counter({'101': 2, '1012': 1}))

    print(is_traversal(my_traver, tree))
    print(f"Traversal Time={len(my_traver)}")

    # for vertex in tree.nodes:
    #     cost1 = compute_signature_cost(vertex, project(vertex, traversal, tree), tree)
    #     cost2 = compute_signature_cost(vertex, project(vertex, my_traver, tree), tree)
    #     print(vertex, cost1, cost2)


