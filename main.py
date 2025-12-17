# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cProfile

import numpy as np
import pandas as pd
from frozendict import frozendict
from tqdm import tqdm
from matplotlib import pyplot as plt

from exploration import tree_cover
from exploration.dfs_explore import dfs_explore
from exploration.picaboo import picaboo
from exploration.baby_giant_step import baby_giant_step
from trees.cocta import cocta_compute_traversal
from trees.configuration import DownArrow, pack_configuration, unpack_configuration
from trees.ncocta import ncocta_compute_traversal
from trees.parallel import parallel_enumerate_signatures
from trees.signature import enumerate_signatures, unpack_signature, project, pack_signature
from trees.table import fpt_compute_traversal, compute_table
from trees.traversal import print_traversal, is_traversal
from trees.tree import print_tree, hard_example_tree, example_tree, jaxsonville_tree, adelphi_tree, \
    random_building_tree, add_floors_to_tree, stretch_halls, increase_room_density, star_tree_example, exploration_tree, \
    chain_tree

from time import time

from plots.graphs import jaxonville_robots_plot, adelphi_plot, adelphi_robots_plot, compare_fpt_cocta, \
    plot_computation_time_graph, adelphi_avg_num_signatures, floor_random_graph_plots, density_random_graph_plots, \
    hall_random_graph_plots

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # tree = example_tree()  # 9
    # tree = hard_example_tree()  # 18
    # tree = jaxsonville_tree(num_floors=7)
    tree = adelphi_tree(num_floors=20)
    # tree = random_building_tree(num_floors=3,
    #                             room_density=0.5,
    #                             max_halls_per_floor=6,
    #                             min_hall_length=2,
    #                             max_hall_length=8)
    # tree = exploration_tree()
    print_tree(tree)

    kk = list(range(5, 100+5, 5))
    coc = []
    exp = []
    for k in tqdm(kk):
        # exp_traversal, _ = dfs_explore(adelphi_tree(num_floors=20), k)
        cocta_traversal = cocta_compute_traversal(adelphi_tree(num_floors=5), num_robots=k)
        # exp.append(len(exp_traversal))
        # coc.append(len(cocta_traversal))
    # plt.plot(kk, np.array(exp)/np.array(coc), label='exp. overhead')
    # plt.plot(kk, 2*np.sqrt(np.array(kk)), label='2 k^1/2')
    # plt.legend()
    # plt.show()

    # print([len(random_building_tree(num_floors=3,
    #                                 room_density=0.5,
    #                                 max_halls_per_floor=6,
    #                                 min_hall_length=2,
    #                                 max_hall_length=8).nodes) for _ in range(100)])
    # traversal = fpt_compute_traversal(tree, num_robots=3, heuristics_on=True)

    # samples = 100
    # load = True
    # hall_random_graph_plots(num_samples=samples, load=False, suffix=f'{samples}_samples')
    # floor_random_graph_plots(num_samples=samples, load=True, suffix=f'{samples}_samples')
    # density_random_graph_plots(num_samples=samples, load=True, suffix=f'{samples}_samples')


    # jaxonville_robots_plot(num_robots=3, num_floors=20)
    # adelphi_robots_plot(num_robots=4, num_floors=8)

    # jax_fpt_df = pd.read_csv('data/jaxonville_fpt.csv')
    # adelphi_fpt_df = pd.read_csv('data/adelphi_fpt_new.csv')
    # compare_fpt_cocta(jax_fpt_df, jaxsonville_tree)
    # compare_fpt_cocta(adelphi_fpt_df, adelphi_tree)

    # plot_computation_time_graph(adelphi_fpt_df)
    # adelphi_avg_num_signatures()


