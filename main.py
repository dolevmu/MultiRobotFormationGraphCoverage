# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cProfile

from frozendict import frozendict
from tqdm import tqdm

from trees.configuration import DownArrow, pack_configuration, unpack_configuration
from trees.parallel import parallel_enumerate_signatures
from trees.signature import enumerate_signatures, unpack_signature, project, pack_signature
from trees.table import fpt_compute_traversal, compute_table
from trees.traversal import print_traversal
from trees.tree import print_tree, hard_example_tree, example_tree, jaxsonville_tree, adelphi_tree

from time import time

from plots.graphs import jaxonville_plot, adelphi_plot, adelphi_robots_plot

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tree = example_tree()  # 9
    # tree = hard_example_tree()  # 18
    # tree = jaxsonville_tree(num_floors=5)
    tree = adelphi_tree(num_floors=2)
    print_tree(tree)

    # 1_006_514
    # sigs = enumerate_signatures('MH6F2', tree, 4, heuristics_on=True)
    # counter = 0
    # for sig in tqdm(sigs):
    #     counter += 1
    # print(counter)

    res = fpt_compute_traversal(tree, num_robots=4, parallel=False, backtrack=True, heuristics_on=True)
    print(res)
    print(len(res))


    # jaxonville_plot(num_robots=3, num_floors=6)
    # adelphi_plot(5)
    # adelphi_robots_plot(num_robots=4, num_floors=5)



