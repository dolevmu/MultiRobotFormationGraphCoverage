from functools import partial
from time import time
from tqdm import tqdm
from typing import Callable

from matplotlib import pyplot as plt
import concurrent.futures
import pandas as pd
import seaborn as sns
from treelib import Tree

from exploration.dfs_explore import dfs_explore
from trees.cocta import cocta_compute_traversal
from trees.signature import enumerate_signatures
from trees.table import fpt_compute_traversal
from trees.tree import adelphi_tree, jaxsonville_tree, random_building_tree, MAX_HALL_LENGTH, MIN_HALL_LENGTH, \
    MAX_HALLS_PER_FLOOR, ROOM_DENSITY, NUM_FLOORS, add_floors_to_tree, increase_room_density, stretch_halls

def exploration_hotel_plot(name: str, hotel_gen: Callable[[int], Tree], num_floors: int = 10, num_robots: int = 100):
    df = pd.read_csv(f'data/exploration_{name}.csv')
    min_floor = round(df["# Floors"].max()) + 1 if len(df) > 0 else 2

    for floor in range(min_floor, num_floors + 1):
        print(f"Floor {floor}/{num_floors}")
        tree = hotel_gen(floor)

        for robots in range(5, num_robots + 5, 5):
            print(f"Robots {robots}/{num_robots}")
            start = time()
            cocta_traversal = cocta_compute_traversal(tree, robots)
            end = time()

            df.loc[len(df)] = ["COCTA", floor, tree.size(), robots, len(cocta_traversal), end - start]

            print()
            print(f"Num robots = {robots}: ")
            print(["COCTA", floor, tree.size(), robots, len(cocta_traversal), end - start])
            print()

            start = time()
            dfs_expl_traversal = dfs_explore(tree, robots)
            end = time()

            df.loc[len(df)] = ["DFS-BGS", floor, tree.size(), robots, len(dfs_expl_traversal), end - start]

            print()
            print(f"Num robots = {robots}: ")
            print(["DFS-BGS", floor, tree.size(), robots, len(dfs_expl_traversal), end - start])
            print()

        # Save progress...
        df.to_csv(f'data/exploration_{name}.csv', index=False)  # Use index=False to avoid saving row indices

    fig, ax1 = plt.subplots(figsize=(10, 8))

    max_robots_df = df[df['# Robots'] == df['# Robots'].max()].copy()

    # Plot Traversal Time
    sns.lineplot(data=max_robots_df, x="# Vertices", y="Traversal Time", style="Algorithm", ax=ax1, color="blue")
    # ax1.plot(max_robots_df["# Vertices"], max_robots_df["Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Vertices", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    sns.lineplot(data=max_robots_df, x="# Vertices", y="Computation Time (sec)", style="Algorithm", ax=ax2, color="red")
    # ax2.plot(max_robots_df["# Vertices"], max_robots_df["Computation Time (hours)"], label="Computation Time (hours)", color="red")
    ax2.set_ylabel("Computation Time (sec)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    # Set up secondary x-axis for # Floors
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())  # Match the x-axis limits with the primary x-axis
    ax3.set_xticks(max_robots_df["# Vertices"])  # Set the same x-axis tick positions as # Vertices
    ax3.set_xticklabels(max_robots_df["# Floors"], fontsize=18)  # Label ticks as # Floors
    ax3.set_xlabel("# Floors", fontsize=20)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    df_restricted = df[df['# Robots'] % 20 == 0].copy()
    sns.lineplot(data=df_restricted.query("Algorithm=='DFS-BGS'"), x='# Vertices', y='Traversal Time', hue='# Robots')
    plt.show()

    df_overhead = df_restricted.query('Algorithm=="DFS-BGS"')
    df_overhead["Overhead"] = (df_overhead["Traversal Time"].to_numpy() / df_restricted.query('Algorithm=="COCTA"')["Traversal Time"].to_numpy())
    sns.lineplot(data=df_overhead, x='# Vertices', y='Overhead', hue='# Robots')
    plt.show()

    df.to_csv(f'data/exploration_{name}.csv', index=False)  # Use index=False to avoid saving row indices


exploration_jaxonville_plot = partial(exploration_hotel_plot, name="jaxonville", hotel_gen=jaxsonville_tree)
exploration_adelphi_plot = partial(exploration_hotel_plot, name="adelphi", hotel_gen=adelphi_tree)


def process_sample(tree, num_robots,
                   hall_length, max_hall_length,
                   room_density, num_floors,
                   num_halls_per_floor, min_hall_length):
    num_vertices = tree.size()
    print(num_vertices)
    num_halls = sum(v.startswith("Branch") for v in tree.nodes) + num_floors
    num_rooms = sum(v.startswith("Room") for v in tree.nodes)

    start = time()
    cocta_traversal = cocta_compute_traversal(tree, num_robots)
    cocta_time = time() - start

    start = time()
    dfs_expl_traversal = dfs_explore(tree, num_robots)
    dfs_expl_time = time() - start

    cocta_result = ["COCTA", num_vertices, num_floors, num_halls, num_rooms, num_halls_per_floor,
                    min_hall_length, hall_length, room_density, len(cocta_traversal), cocta_time]

    dfs_expl_result = ["DFS-BGS", num_vertices, num_floors, num_halls, num_rooms, num_halls_per_floor,
                       min_hall_length, hall_length, room_density, len(dfs_expl_traversal), dfs_expl_time]

    return {"COCTA": cocta_result, "DFS-BGS": dfs_expl_result}


def density_random_graph_plots(num_samples: int = 100,
                               load: bool = True,
                               suffix: str = ''):
    num_floors = NUM_FLOORS
    num_robots = 40

    num_halls_per_floor = MAX_HALLS_PER_FLOOR
    min_hall_length = MIN_HALL_LENGTH
    max_hall_length = MAX_HALL_LENGTH

    trees = [random_building_tree() for _ in range(num_samples)]

    if load:
        density_df = pd.read_csv(f'data/density_df{suffix}.csv')

    else:
        density_df = pd.DataFrame(
            columns=["# Vertices", "# Floors", "# Halls", "# Rooms", "# Halls per floor", "Min. hall length",
                     "Max. hall length", "Room Density", "Traversal Time", "% Saved Time", "Computation Time (sec)",
                     "Computation Time (min)"])

        for room_density in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for sample in range(num_samples):
                print(f"Room Density = {room_density}")

                tree = trees[sample]

                num_vertices = tree.size()
                num_halls = sum(v.startswith("Branch") for v in tree.nodes) + NUM_FLOORS
                num_rooms = sum(v.startswith("Room") for v in tree.nodes)

                pacman_time = time()
                try:
                    pacman_traversal = fpt_compute_traversal(tree, num_robots, heuristics_on=True, backtrack=True,
                                                             max_sig_length=9)
                except:
                    print("Failed")
                pacman_time = time() - pacman_time

                # cocta_time = time()
                try:
                    cocta_traversal = cocta_compute_traversal(tree, num_robots)
                except:
                    print("Failed")
                    continue
                # cocta_time = time() - cocta_time

                saved_time = (len(cocta_traversal) - len(pacman_traversal)) / len(cocta_traversal) * 100

                density_df.loc[len(density_df)] = [num_vertices, num_floors, num_halls, num_rooms, num_halls_per_floor,
                                               min_hall_length, max_hall_length, room_density,
                                               len(pacman_traversal), saved_time, pacman_time, pacman_time / 60]

                # Update tree for next iteration
                trees[sample] = increase_room_density(tree, room_density_to_add=0.2)

            # Save progress...
            density_df.to_csv(f'data/density_df{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    sns.lineplot(
        data=density_df, x="Room Density", y="Traversal Time",
        ax=ax1, color="blue", label="Traversal Time",
        estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    # ax1.plot(density_df["Room Density"], density_df.groupby("Room Density")["Traversal Time"].transform('mean'),
    #          label="Traversal Time", color="blue")
    ax1.set_xlabel("Room Density", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    sns.lineplot(
        data=density_df, x="Room Density", y="% Saved Time",
        ax=ax2, color="red", label="% Saved Time",
        estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    # ax2.plot(density_df["Room Density"], density_df.groupby("Room Density")["% Saved Time"].transform('mean'), label="% Saved Time",
    #          color="red")
    ax2.set_ylabel("% Saved Time", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    density_df.to_csv('data/floor_df.csv', index=False)  # Use index=False to avoid saving row indices


def exploration_hall_random_graph_plots(num_samples: int = 100,
                            load: bool = False,
                            suffix: str = '',
                            num_robots: int = 30,
                            num_floors: int = 10,
                            max_halls_per_floor: int = 10,
                            min_hall_length: int = 5,
                            max_hall_length: int = 50,
                            room_density: int = 0.3
):
    trees = [random_building_tree(num_floors=num_floors,
                                  max_halls_per_floor=max_halls_per_floor,
                                  min_hall_length=min_hall_length,
                                  max_hall_length=max_hall_length,
                                  room_density=room_density) for _ in range(num_samples)]

    if load:
        df = pd.read_csv(f'data/exploration_hall_df_{suffix}.csv')

    else:
        df = pd.DataFrame(columns=["Algorithm", "# Vertices", "# Floors", "# Halls", "# Rooms", "# Halls per floor", "Min. hall length", "Max. hall length", "Room Density", "Traversal Time", "Computation Time (sec)"])

        for hall_length in range(min_hall_length, max_hall_length + 5, 5):
            print(f"Hall Length {hall_length}/{max_hall_length}")
            for sample in tqdm(range(num_samples)):
                result = process_sample(trees[sample], num_robots,
                                        hall_length, max_hall_length,
                                        room_density, num_floors,
                                        max_halls_per_floor, min_hall_length)
                df.loc[len(df)] = result["COCTA"]
                df.loc[len(df)] = result["DFS-BGS"]
                trees[sample] = stretch_halls(trees[sample], hall_length_to_add=1, room_density=room_density)

        # Save progress...
        df.to_csv(f'data/exploration_hall_df{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    fig, ax1 = plt.subplots(figsize=(10, 8))
    max_hall_length_df = df.groupby(["Algorithm", "Max. hall length"])[["Traversal Time", "Computation Time (sec)"]].transform('mean').reset_index()

    # Plot Traversal Time
    sns.lineplot(data=max_hall_length_df, x="Max. hall length", y="Traversal Time", style="Algorithm", ax=ax1, color="blue")
    # ax1.plot(max_robots_df["# Vertices"], max_robots_df["Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Vertices", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    sns.lineplot(data=max_hall_length_df, x="Max. hall length", y="Computation Time (sec)", style="Algorithm", ax=ax2, color="red")
    # ax2.plot(max_robots_df["# Vertices"], max_robots_df["Computation Time (hours)"], label="Computation Time (hours)", color="red")
    ax2.set_ylabel("Computation Time (sec)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    # # Set up secondary x-axis for # Floors
    # ax3 = ax1.twiny()
    # ax3.set_xlim(ax1.get_xlim())  # Match the x-axis limits with the primary x-axis
    # ax3.set_xticks(max_hall_length_df["# Vertices"])  # Set the same x-axis tick positions as # Vertices
    # ax3.set_xticklabels(max_hall_length_df["# Floors"], fontsize=18)  # Label ticks as # Floors
    # ax3.set_xlabel("# Floors", fontsize=20)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    df.to_csv(f'data/exploration_hall_df{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

