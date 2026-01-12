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
from trees.tree import adelphi_tree, jaxsonville_tree, random_building_tree, random_recursive_tree, uniform_random_tree, MAX_HALL_LENGTH, MIN_HALL_LENGTH, \
    MAX_HALLS_PER_FLOOR, ROOM_DENSITY, NUM_FLOORS, add_floors_to_tree, increase_room_density, stretch_halls
import numpy as np

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
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    sns.lineplot(data=max_robots_df, x="# Vertices", y="Computation Time (sec)", style="Algorithm", ax=ax2, color="red")
    # ax2.plot(max_robots_df["# Vertices"], max_robots_df["Computation Time (hours)"], label="Computation Time (hours)", color="red")
    ax2.set_ylabel("Computation Time (sec)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=20)

    # Set up secondary x-axis for # Floors
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())  # Match the x-axis limits with the primary x-axis
    ax3.set_xticks(max_robots_df["# Vertices"])  # Set the same x-axis tick positions as # Vertices
    ax3.set_xticklabels(max_robots_df["# Floors"], fontsize=20)  # Label ticks as # Floors
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


def exploration_density_random_graph_plots(num_samples: int = 100,
                                           load: bool = True,
                                           suffix: str = '',
                                           num_robots: int = 30,
                                           num_floors: int = 8,
                                           max_halls_per_floor: int = 8,
                                           min_hall_length: int = 5,
                                           max_hall_length: int = 40,
                                           ):

    room_density = 0  # Will be increased incrementally

    trees = [random_building_tree(num_floors=num_floors,
                                  max_halls_per_floor=max_halls_per_floor,
                                  min_hall_length=min_hall_length,
                                  max_hall_length=max_hall_length,
                                  room_density=room_density) for _ in range(num_samples)]

    if load:
        density_df = pd.read_csv(f'data/exploration_density_df_{suffix}.csv')

    else:
        density_df = pd.DataFrame(
            columns=["Algorithm", "# Vertices", "# Floors", "# Halls", "# Rooms", "# Halls per floor", "Min. hall length",
                     "Max. hall length", "Room Density", "Traversal Time", "Computation Time (sec)"])

        for room_density in tqdm([0, 0.2, 0.4, 0.6, 0.8, 1.0], "Room Density"):
            for sample in tqdm(range(num_samples), "Sample #"):
                # print(f"Room Density = {room_density}")

                tree = trees[sample]

                num_vertices = tree.size()
                num_halls = sum(v.startswith("Branch") for v in tree.nodes) + NUM_FLOORS
                num_rooms = sum(v.startswith("Room") for v in tree.nodes)


                cocta_time = time()
                cocta_traversal = cocta_compute_traversal(tree, num_robots)
                cocta_time = time() - cocta_time

                cocta_result = ["COCTA", num_vertices, num_floors, num_halls, num_rooms, max_halls_per_floor,
                                min_hall_length, max_hall_length, room_density, len(cocta_traversal), cocta_time]

                density_df.loc[len(density_df)] = cocta_result

                dfs_bgs_time = time()
                dfs_bgs_traversal = dfs_explore(tree, num_robots)
                dfs_bgs_time = time() - dfs_bgs_time

                dfs_bgs_result = ["DFS-BGS", num_vertices, num_floors, num_halls, num_rooms, max_halls_per_floor,
                                min_hall_length, max_hall_length, room_density, len(dfs_bgs_traversal), dfs_bgs_time]

                density_df.loc[len(density_df)] = dfs_bgs_result

                # Update tree for next iteration
                trees[sample] = increase_room_density(tree, room_density_to_add=0.2)

            # Save progress...
            density_df.to_csv(f'data/exploration_density_df_{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    # fig, ax1 = plt.subplots(figsize=(10, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    # Plot Traversal Time
    sns.lineplot(
        data=density_df, x="Room Density", y="Traversal Time",
        ax=ax1, hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )

    ax1.set_xlabel("Room Density", fontsize=20)
    ax1.set_ylabel("Traversal Time", fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    # Set up secondary y-axis for Computation Time
    # ax2 = ax1.twinx()
    sns.lineplot(
        data=density_df, x="Room Density", y="Computation Time (sec)",
        ax=ax2, hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    ax1.set_xlabel("Room Density", fontsize=20)
    ax2.set_ylabel("Computation Time (sec)", fontsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    # Plot Traversal Time
    keys = [
        "# Vertices",
        "# Floors",
        "# Halls",
        "# Rooms",
        "# Halls per floor",
        "Min. hall length",
        "Max. hall length",
        "Room Density"
    ]
    pivot_df = density_df.pivot(index=keys, values=["Traversal Time"], columns=["Algorithm"])
    pivot_df["Overhead"] = pivot_df["DFS-BGS"] / pivot_df["COCTA"]
    pivot_df.reset_index()
    sns.lineplot(
        data=pivot_df, x="Room Density", y="Overhead", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    plt.xlabel("Room Density", fontsize=20)
    plt.ylabel("Traversal Time", fontsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.show()

    density_df.to_csv(f'data/exploration_density_df_{suffix}.csv', index=False)  # Use index=False to avoid saving row indices


def exploration_hall_random_graph_plots(num_samples: int = 100,
                            load: bool = False,
                            suffix: str = '',
                            num_robots: int = 30,
                            num_floors: int = 8,
                            max_halls_per_floor: int = 8,
                            min_hall_length: int = 5,
                            max_hall_length: int = 40,
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
            df.to_csv(f'data/exploration_hall_df_{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    # fig, ax1 = plt.subplots(figsize=(10, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    # Plot Traversal Time
    sns.lineplot(data=df, x="Max. hall length", y="Traversal Time",
                 hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}, ax=ax1)
    ax1.set_xlabel("# Max. hall length", fontsize=20)
    ax1.set_ylabel("Traversal Time", fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.set_yscale('log')
    ax1.legend()

    # Set up secondary y-axis for Computation Time
    # ax2 = ax1.twinx()
    sns.lineplot(data=df, x="Max. hall length", y="Computation Time (sec)",
                 hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}, ax=ax2)
    ax1.set_xlabel("# Max. hall length", fontsize=20)
    ax2.set_ylabel("Computation Time (sec)", fontsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.set_yscale('log')
    ax2.legend()

    # # Set up secondary x-axis for # Floors
    # ax3 = ax1.twiny()
    # ax3.set_xlim(ax1.get_xlim())  # Match the x-axis limits with the primary x-axis
    # ax3.set_xticks(max_hall_length_df["# Vertices"])  # Set the same x-axis tick positions as # Vertices
    # ax3.set_xticklabels(max_hall_length_df["# Floors"], fontsize=20)  # Label ticks as # Floors
    # ax3.set_xlabel("# Floors", fontsize=20)

    # # Collect legend handles and labels from both axes
    # handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    #
    # # Remove automatic legends
    # ax1.legend_.remove()
    # ax2.legend_.remove()

    # # Create a combined legend
    # ax1.legend(
    #     handles1 + handles2,
    #     labels1 + labels2,
    #     loc="upper left",
    #     fontsize=20
    # )

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    df.to_csv(f'data/exploration_hall_df_{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

def exploration_floor_random_graph_plots(num_samples: int = 100,
                                         load: bool = True,
                                         suffix: str = '',
                                         num_robots: int = 30,
                                         min_num_floors: int = 4,
                                         max_num_floors: int = 10,
                                         max_halls_per_floor: int = 8,
                                         min_hall_length: int = 5,
                                         max_hall_length: int = 40,
                                         room_density: int = 0.3):
    max_floors_to_add = max_num_floors - min_num_floors

    trees = [random_building_tree(num_floors=min_num_floors,
                                  max_halls_per_floor=max_halls_per_floor,
                                  min_hall_length=min_hall_length,
                                  max_hall_length=max_hall_length,
                                  room_density=room_density) for _ in range(num_samples)]

    if load:
        floor_df = pd.read_csv(f'data/exploration_floor_df_{suffix}.csv')

    else:
        floor_df = pd.DataFrame(columns=["Algorithm", "# Vertices", "# Floors", "# Halls", "# Rooms", "# Halls per floor", "Min. hall length", "Max. hall length", "Room Density", "Traversal Time", "Computation Time (sec)"])
        num_floors = min_num_floors
        for _ in tqdm(range(max_floors_to_add), desc=f"Floor {num_floors}/{max_num_floors}", leave=False, position=0):
            sample = 0
            for _ in tqdm(range(num_samples), desc=f"Sample {sample}/{num_samples}", leave=False, position=1):
                tree = trees[sample]

                num_vertices = tree.size()
                num_halls = sum(v.startswith("Branch") for v in tree.nodes) + NUM_FLOORS
                num_rooms = sum(v.startswith("Room") for v in tree.nodes)


                cocta_time = time()
                cocta_traversal = cocta_compute_traversal(tree, num_robots)
                cocta_time = time() - cocta_time

                cocta_result = ["COCTA", num_vertices, num_floors, num_halls, num_rooms, max_halls_per_floor,
                                min_hall_length, max_hall_length, room_density, len(cocta_traversal), cocta_time]

                floor_df.loc[len(floor_df)] = cocta_result

                dfs_bgs_time = time()
                dfs_bgs_traversal = dfs_explore(tree, num_robots)
                dfs_bgs_time = time() - dfs_bgs_time

                dfs_bgs_result = ["DFS-BGS", num_vertices, num_floors, num_halls, num_rooms, max_halls_per_floor,
                                min_hall_length, max_hall_length, room_density, len(dfs_bgs_traversal), dfs_bgs_time]

                floor_df.loc[len(floor_df)] = dfs_bgs_result

                # Update tree for next iteration
                trees[sample] = add_floors_to_tree(tree, 1, num_floors=num_floors)
                sample += 1
            num_floors += 1  # Update number of floors

            # Save progress...
            floor_df.to_csv(f'data/exploration_floor_df_{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    # fig, ax1 = plt.subplots(figsize=(10, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    # Plot Traversal Time
    sns.lineplot(
        data=floor_df, x="# Floors", y="Traversal Time",
        ax=ax1, hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )

    ax1.set_xlabel("# Floors", fontsize=20)
    ax1.set_ylabel("Traversal Time", fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', labelsize=20)

    # Set up secondary y-axis for Computation Time
    # ax2 = ax1.twinx()
    sns.lineplot(
        data=floor_df, x="# Floors", y="Computation Time (sec)",
        ax=ax2, hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    ax1.set_xlabel("# Floors", fontsize=20)
    ax2.set_ylabel("Computation Time (sec)", fontsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', labelsize=20)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    floor_df.to_csv(f'data/exploration_floor_df_{suffix}.csv', index=False)  # Use index=False to avoid saving row indices


def exploration_rrt_speedup_plot(num_samples: int = 10,
                                  n: int = 10_000,
                                  min_robots: int = 3,
                                  max_robots: int = 30,
                                  robot_step: int = 3,
                                  algorithms: list = None,
                                  tree_type: str = "uniform",
                                  load: bool = False,
                                  save_path: str = None):
    """
    Benchmark DFS-BGS and/or COCTA on random trees, plotting speedup vs number of robots.

    Speedup is defined as n / len(traversal), i.e., how many nodes covered per traversal step.

    :param num_samples: Number of random trees to generate for averaging
    :param n: Number of nodes in each tree
    :param min_robots: Minimum number of robots to test
    :param max_robots: Maximum number of robots to test
    :param robot_step: Step size for robot count (e.g., 3 means test 3, 6, 9, ...)
    :param algorithms: List of algorithms to run, e.g., ["DFS-BGS"], ["COCTA"], or ["DFS-BGS", "COCTA"]
    :param tree_type: Type of random tree: "uniform" (Prüfer) or "recursive" (RRT)
    :param load: If True, load existing results from CSV
    :param save_path: Path to save the plot PNG (default: data/exploration_{tree_type}_{num_samples}_samples.png)
    """
    if algorithms is None:
        algorithms = ["DFS-BGS", "COCTA"]

    tree_gen = uniform_random_tree if tree_type == "uniform" else random_recursive_tree

    if save_path is None:
        alg_suffix = "_".join(algorithms).lower().replace("-", "")
        save_path = f'data/exploration_{tree_type}_{alg_suffix}_{num_samples}_samples.png'
    csv_path = save_path.replace('.png', '.csv')

    if load:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Algorithm", "Sample", "# Vertices", "# Robots",
                                   "Traversal Time", "Speedup", "Computation Time (sec)"])

        # Generate trees once
        trees = [tree_gen(n, seed=i) for i in tqdm(range(num_samples), desc="Generating trees")]

        for k in tqdm(range(min_robots, max_robots + 1, robot_step), desc="# Robots"):
            for sample_idx, tree in enumerate(trees):
                if "COCTA" in algorithms:
                    start = time()
                    cocta_traversal = cocta_compute_traversal(tree, k)
                    cocta_time = time() - start
                    cocta_speedup = n / len(cocta_traversal)
                    df.loc[len(df)] = ["COCTA", sample_idx, n, k, len(cocta_traversal), cocta_speedup, cocta_time]

                if "DFS-BGS" in algorithms:
                    start = time()
                    dfs_traversal = dfs_explore(tree, k)
                    dfs_time = time() - start
                    dfs_speedup = n / len(dfs_traversal)
                    df.loc[len(df)] = ["DFS-BGS", sample_idx, n, k, len(dfs_traversal), dfs_speedup, dfs_time]

            # Save progress
            df.to_csv(csv_path, index=False)

    # Plot speedup
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.lineplot(
        data=df, x="# Robots", y="Speedup",
        hue="Algorithm", estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}, ax=ax
    )

    ax.set_xlabel("# Robots (k)", fontsize=16)
    ax.set_ylabel("Speedup (n / traversal length)", fontsize=16)
    ax.set_title(f"Speedup on Random Trees (n={n:,}, {num_samples} samples)", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Results saved to {csv_path}")
    print(f"Plot saved to {save_path}")

    return df


def pref_attach_speedup_vs_k(csv_path: str = 'data/pref_attach_speedup.csv',
                              n_values: list = None,
                              save_path: str = None):
    """
    Plot speedup vs k for fixed n values on preferential attachment trees.
    Compares COCTA and BGS (k^1/3) algorithms.
    """
    df = pd.read_csv(csv_path)

    if n_values is None:
        n_values = [1000, 5000]

    fig, axes = plt.subplots(1, len(n_values), figsize=(7 * len(n_values), 6))
    if len(n_values) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        df_n = df[df['n'] == n].copy()

        # Plot COCTA speedup
        cocta_data = df_n[['k', 'cocta_speedup']].dropna()
        ax.plot(cocta_data['k'], cocta_data['cocta_speedup'], 'o-', label='COCTA', markersize=8, linewidth=2)

        # Plot BGS speedup
        bgs_data = df_n[['k', 'bgs_speedup']].dropna()
        ax.plot(bgs_data['k'], bgs_data['bgs_speedup'], 's-', label='BGS (Δ=k^⅓)', markersize=8, linewidth=2)

        # Add reference line k/depth
        depth = df_n['depth'].iloc[0]
        k_range = np.array(sorted(df_n['k'].unique()))
        ax.plot(k_range, k_range / depth, '--', color='gray', alpha=0.7, label=f'k/D (D={depth})')

        ax.set_xlabel('Number of Robots (k)', fontsize=14)
        ax.set_ylabel('Speedup (n / traversal length)', fontsize=14)
        ax.set_title(f'Preferential Attachment Tree (n={n:,})', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return df


def pref_attach_speedup_vs_n(csv_path: str = 'data/pref_attach_speedup.csv',
                              k_values: list = None,
                              save_path: str = None):
    """
    Plot speedup vs n for fixed k values on preferential attachment trees.
    Compares COCTA and BGS (k^1/3) algorithms.
    """
    df = pd.read_csv(csv_path)

    if k_values is None:
        k_values = [100, 400]

    fig, axes = plt.subplots(1, len(k_values), figsize=(7 * len(k_values), 6))
    if len(k_values) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        df_k = df[df['k'] == k].copy()

        # Plot COCTA speedup
        cocta_data = df_k[['n', 'cocta_speedup']].dropna()
        ax.plot(cocta_data['n'], cocta_data['cocta_speedup'], 'o-', label='COCTA', markersize=8, linewidth=2)

        # Plot BGS speedup
        bgs_data = df_k[['n', 'bgs_speedup']].dropna()
        ax.plot(bgs_data['n'], bgs_data['bgs_speedup'], 's-', label='BGS (Δ=k^⅓)', markersize=8, linewidth=2)

        ax.set_xlabel('Number of Vertices (n)', fontsize=14)
        ax.set_ylabel('Speedup (n / traversal length)', fontsize=14)
        ax.set_title(f'Preferential Attachment Tree (k={k})', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Mark n=k line
        ax.axvline(x=k, color='red', linestyle=':', alpha=0.5, label=f'n=k={k}')

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return df


def pref_attach_overhead_plots(csv_path: str = 'data/pref_attach_speedup.csv',
                                save_path: str = None):
    """
    Plot overhead (COCTA speedup / BGS speedup) for preferential attachment trees.
    Values > 1 mean COCTA is better, < 1 means BGS is better.
    """
    df = pd.read_csv(csv_path)

    # Calculate overhead where both values exist
    df['overhead'] = df['cocta_speedup'] / df['bgs_speedup']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Overhead vs k for different n values
    for n in sorted(df['n'].unique()):
        df_n = df[df['n'] == n].dropna(subset=['overhead'])
        if len(df_n) > 0:
            ax1.plot(df_n['k'], df_n['overhead'], 'o-', label=f'n={n}', markersize=6)

    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal performance')
    ax1.set_xlabel('Number of Robots (k)', fontsize=14)
    ax1.set_ylabel('Overhead (COCTA speedup / BGS speedup)', fontsize=14)
    ax1.set_title('COCTA vs BGS Overhead by k', fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Overhead vs n for different k values
    for k in [100, 200, 400, 750]:
        df_k = df[df['k'] == k].dropna(subset=['overhead'])
        if len(df_k) > 0:
            ax2.plot(df_k['n'], df_k['overhead'], 'o-', label=f'k={k}', markersize=6)

    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal performance')
    ax2.set_xlabel('Number of Vertices (n)', fontsize=14)
    ax2.set_ylabel('Overhead (COCTA speedup / BGS speedup)', fontsize=14)
    ax2.set_title('COCTA vs BGS Overhead by n', fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    # Print summary
    valid_overhead = df['overhead'].dropna()
    print(f"\nOverhead Summary:")
    print(f"  Min: {valid_overhead.min():.3f}")
    print(f"  Max: {valid_overhead.max():.3f}")
    print(f"  Mean: {valid_overhead.mean():.3f}")
    print(f"  Cases where BGS > COCTA (overhead < 1): {(valid_overhead < 1).sum()} / {len(valid_overhead)}")

    return df
