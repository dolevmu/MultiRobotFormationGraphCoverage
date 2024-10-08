# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter

from frozendict import frozendict

from trees.configuration import DownArrow, UpArrow
from trees.signature import pack_signature, unpack_signature
from trees.table import fpt_compute_traversal
from trees.traversal import print_traversal
from trees.tree import print_tree, hard_example_tree, example_tree, jaxsonville_tree

from time import time

BACK_TRACK = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tree = example_tree()
    tree = hard_example_tree()
    # tree = jaxsonville_tree()
    print_tree(tree)

    start = time()
    traversal = fpt_compute_traversal(tree, 3, backtrack=BACK_TRACK)
    end = time()

    print(f"Computation Time={end - start:.2f}")
    if BACK_TRACK:
        print(f"Traversal Time={len(traversal)}, Tree Size={tree.size()}")
        print_traversal(traversal)
    else:
        print(f"Traversal Time={traversal}, Tree Size={tree.size()}")
