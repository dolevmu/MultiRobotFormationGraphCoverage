# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from trees.table import fpt_compute_traversal
from trees.traversal import print_traversal
from trees.tree import example_tree, print_tree
import numpy as np

from time import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tree = example_tree()
    print_tree(tree)

    start = time()
    traversal = fpt_compute_traversal(tree, 3)
    end = time()

    print(f"Traversal Time={len(traversal)}, Tree Size={tree.size()}")
    print(f"Computation Time={end-start:.2f}")
    print_traversal(traversal)
