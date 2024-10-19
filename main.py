# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from trees.signature import enumerate_signatures, unpack_signature
from trees.table import fpt_compute_traversal
from trees.traversal import print_traversal
from trees.tree import print_tree, hard_example_tree, example_tree, jaxsonville_tree, adelphi_tree

from time import time

from plots.graphs import jaxonville_plot, adelphi_plot, adelphi_robots_plot

BACK_TRACK = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tree = example_tree()
    # tree = hard_example_tree()
    # tree = jaxsonville_tree(num_floors=5)
    tree = adelphi_tree(num_floors=5)
    print_tree(tree)

    # sigs = enumerate_signatures('MH6F2', tree, 2, heuristics_on=False)

    # jaxonville_plot()
    adelphi_plot(5)
    # adelphi_robots_plot(4)


