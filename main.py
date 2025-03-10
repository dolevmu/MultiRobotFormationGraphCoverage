# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cProfile

import pandas as pd
from frozendict import frozendict
from tqdm import tqdm

from trees.cocta import cocta_compute_traversal
from trees.configuration import DownArrow, pack_configuration, unpack_configuration
from trees.ncocta import ncocta_compute_traversal
from trees.parallel import parallel_enumerate_signatures
from trees.signature import enumerate_signatures, unpack_signature, project, pack_signature
from trees.table import fpt_compute_traversal, compute_table
from trees.traversal import print_traversal, is_traversal
from trees.tree import print_tree, hard_example_tree, example_tree, jaxsonville_tree, adelphi_tree

from time import time

from plots.graphs import jaxonville_robots_plot, adelphi_plot, adelphi_robots_plot, compare_fpt_cocta, \
    plot_computation_time_graph, adelphi_avg_num_signatures

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tree = example_tree()  # 9
    # tree = hard_example_tree()  # 18
    # tree = jaxsonville_tree(num_floors=4)
    tree = adelphi_tree(num_floors=5)
    print_tree(tree)


    # 1_006_514
    # sigs = enumerate_signatures('EL5', tree, 3, heuristics_on=True)
    # counter = 0
    # for sig in tqdm(sigs):
    #     counter += 1
    # print(counter)

    # for i in range(2, 10):
    #     tree = adelphi_tree(num_floors=i)
    #     fpt = fpt_compute_traversal(tree, num_robots=3, heuristics_on=True)
    #     cocta = cocta_compute_traversal(tree, num_robots = 3)
    #     print(f"Num Floors = {i}, FPT = {len(fpt)}, COCTA = {len(cocta)}")


    # jaxonville_robots_plot(num_robots=3, num_floors=20)
    # adelphi_plot(5)
    # adelphi_robots_plot(num_robots=4, num_floors=8)

    # traversal = ncocta_compute_traversal(tree, num_robots=3, hh=[2,1])
    # traversal = cocta_compute_traversal(tree, num_robots=4, hh=[3,2,1])
    # print(is_traversal(traversal, tree), len(traversal))
    # print(traversal)

    # jax_fpt_df = pd.read_csv('data/jaxonville_fpt.csv')
    adelphi_fpt_df = pd.read_csv('data/adelphi_fpt_new.csv')
    # compare_fpt_cocta(jax_fpt_df, jaxsonville_tree)
    # compare_fpt_cocta(adelphi_fpt_df, adelphi_tree)

    plot_computation_time_graph(adelphi_fpt_df)
    # adelphi_avg_num_signatures()


