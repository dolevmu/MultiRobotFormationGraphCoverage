# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from trees.table import fpt_compute_traversal
from trees.traversal import print_traversal
from trees.tree import print_tree, hard_example_tree, example_tree, jaxsonville_tree, adelphi_tree

from time import time

from plots.graphs import jaxonville_plot

BACK_TRACK = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tree = example_tree()
    # tree = hard_example_tree()
    # tree = jaxsonville_tree(num_floors=5)
    tree = adelphi_tree(num_floors=3)
    print_tree(tree)

    start = time()
    traversal = fpt_compute_traversal(tree, 2, backtrack=BACK_TRACK, heuristics_on=False)
    end = time()

    print(f"Computation Time={end - start:.2f}")
    if BACK_TRACK:
        print(f"Traversal Time={len(traversal)}, Tree Size={tree.size()}")
        print_traversal(traversal)
    else:
        print(f"Traversal Time={traversal}, Tree Size={tree.size()}")

    # jaxonville_plot()
