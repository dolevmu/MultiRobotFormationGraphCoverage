# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cProfile

from frozendict import frozendict
from tqdm import tqdm

from trees.configuration import DownArrow
from trees.signature import enumerate_signatures, unpack_signature, project
from trees.table import fpt_compute_traversal, compute_table
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
    tree = adelphi_tree(num_floors=3)
    print_tree(tree)

    # cProfile.run("enumerate_signatures('MH6F3', tree, 2, heuristics_on=False)")
    # cProfile.run("enumerate_signatures('MH6F3', tree, 2, heuristics_on=True)")

    # cProfile.run("compute_table('EL1', tree, 2, False, heuristics_on=False)")
    # cProfile.run("compute_table('EL1', tree, 2, False, heuristics_on=True)")

    # jaxonville_plot()
    adelphi_plot(5)
    # adelphi_robots_plot(4)


