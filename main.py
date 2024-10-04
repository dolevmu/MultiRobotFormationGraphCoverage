# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter

from trees.signature import project, compute_signature_cost
from trees.table import fpt_compute_traversal
from trees.traversal import print_traversal, is_traversal
from trees.tree import hard_example_tree, example_tree, print_tree, jaxsonville_tree

from time import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tree = hard_example_tree()
    tree = jaxsonville_tree()
    print_tree(tree)

    # start = time()
    # traversal = fpt_compute_traversal(tree, 3)
    # end = time()
    #
    # print(f"Traversal Time={len(traversal)}, Tree Size={tree.size()}")
    # print(f"Computation Time={end-start:.2f}")
    # print_traversal(traversal)
