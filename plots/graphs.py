from time import time
from tqdm import tqdm

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from trees.cocta import cocta_compute_traversal
from trees.table import fpt_compute_traversal
from trees.tree import adelphi_tree, jaxsonville_tree


def jaxonville_robots_plot(num_floors: int = 6, num_robots: int = 3, max_sig_length=9):
    jax_df = pd.read_csv('data/jaxonville_fpt.csv')

    for floor in range(round(jax_df["# Floors"].max()) + 1, num_floors + 1):
        print(f"Floor {floor}/{num_floors}")
        tree = jaxsonville_tree(num_floors=floor)

        for robots in range(1, num_robots + 1):
            print(f"Robots {robots}/{num_robots}")
            start = time()
            traversal = fpt_compute_traversal(tree, robots, heuristics_on=True, backtrack=True, max_sig_length=max_sig_length)
            end = time()

            jax_df.loc[len(jax_df)] = [floor, tree.size(), robots, len(traversal), end - start, (end - start) / 3600]
            print()
            print(f"Num robots = {robots}: ")
            print([floor, tree.size(), robots, len(traversal), end - start])
            print()

        # Save progress...
        jax_df.to_csv('data/adelphi_fpt.csv', index=False)  # Use index=False to avoid saving row indices


    # Convert computation time from seconds to hours
    jax_df["Computation Time (hours)"] = jax_df["Computation Time (sec)"] / 3600

    fig, ax1 = plt.subplots(figsize=(10, 8))

    max_robots_df = jax_df[jax_df['# Robots'] == jax_df['# Robots'].max()].copy()

    # Plot Traversal Time
    ax1.plot(max_robots_df["# Vertices"], max_robots_df["Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Vertices", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    ax2.plot(max_robots_df["# Vertices"], max_robots_df["Computation Time (hours)"], label="Computation Time (hours)", color="red")
    ax2.set_ylabel("Computation Time (hours)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    # Set up secondary x-axis for # Floors
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())  # Match the x-axis limits with the primary x-axis
    ax3.set_xticks(max_robots_df["# Vertices"])  # Set the same x-axis tick positions as # Vertices
    ax3.set_xticklabels(max_robots_df["# Floors"], fontsize=18)  # Label ticks as # Floors
    ax3.set_xlabel("# Floors", fontsize=20)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    sns.lineplot(data=jax_df, x='# Vertices', y='Traversal Time', hue='# Robots')
    plt.show()

    jax_df.to_csv('data/jaxonville_fpt.csv', index=False)  # Use index=False to avoid saving row indices



def adelphi_plot(num_floors: int):  # Adelphi Hotel, Melbourne

    # adelphi_df = pd.DataFrame(data={"# Floors": [],
    #                                 "# Vertices": [],
    #                                 "Traversal Time": [],
    #                                 "Computation Time (sec)": [],
    #                                 "Heuristics": []})

    adelphi_df = pd.DataFrame(data={"# Floors": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                                    "# Vertices": [2, 2, 19, 19, 36, 36, 53, 53, 70, 70],
                                    "Traversal Time": [2, 2,
                                                       19, 19,
                                                       45, 45,
                                                       71, 71,
                                                       97, 97],
                                    "Computation Time (sec)": [0.04491138458251953, 0.041311025619506836,
                                                               457.55689001083374, 110.27886056900024,
                                                               1583.7888889312744, 919.4530813694,
                                                               3223.60542011261, 2899.9116559028625,
                                                               5216.914257287979, 6214.148998498917],
                                    "Heuristics": ["Off", "On", "Off", "On", "Off", "On", "Off", "On", "Off", "On"]})

    computed = len(adelphi_df) // 2
    for floor in range(computed + 1, num_floors + 1):
        print(f"Floor {floor}/{num_floors}")
        tree = adelphi_tree(num_floors=floor)

        precise_start = time()
        precise_traversal = fpt_compute_traversal(tree, 2, heuristics_on=False)
        precise_end = time()

        print(f"Precise: {precise_end - precise_start}, {len(precise_traversal)}")

        heuristic_start = time()
        heuristic_traversal = fpt_compute_traversal(tree, 2, heuristics_on=True)
        heuristic_end = time()

        print(f"Heuristic: {heuristic_end-heuristic_start}, {len(heuristic_traversal)}")

        adelphi_df.loc[len(adelphi_df)] = [floor, tree.size(), len(precise_traversal), precise_end - precise_start, "Off"]
        adelphi_df.loc[len(adelphi_df)] = [floor, tree.size(), len(heuristic_traversal), heuristic_end - heuristic_start, "On"]

        print([floor, tree.size(), len(precise_traversal), precise_end - precise_start, "Off"])
        print([floor, tree.size(), len(heuristic_traversal), heuristic_end - heuristic_start, "On"])

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Plot traversal time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=adelphi_df, x="# Floors", y="Traversal Time", hue="Heuristics", marker="o")
    plt.xlabel("# Floors", fontsize=14)
    plt.ylabel("Traversal Time", fontsize=14)
    plt.title("Traversal Time with and without Heuristics", fontsize=16)
    plt.legend(title="Heuristics")
    plt.show()

    # Plot computation time
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=adelphi_df, x="# Floors", y="Computation Time (sec)", hue="Heuristics", marker="o")
    plt.xlabel("# Floors", fontsize=20)
    plt.ylabel("Computation Time (sec)", fontsize=20)
    # plt.title("Computation Time with and without Heuristics", fontsize=24)
    plt.legend()  # title="Heuristics")
    plt.show()

def adelphi_robots_plot(num_robots: int, num_floors: int = 5, max_sig_length: int = 8):
    # adelphi_df = pd.DataFrame(data={"# Floors": [],
    #                                 "# Vertices": [],
    #                                 "# Robots": [],
    #                                 "Traversal Time": [],
    #                                 "Computation Time (sec)": []})

    # adelphi_df = pd.DataFrame(data={"# Floors": [5, 5, 5, 5],
    #                                 "# Vertices": [70, 70, 70, 70],
    #                                 "# Robots": [1, 2, 3, 4],
    #                                 "Traversal Time": [125, 97, 87, 84],
    #                                 "Computation Time (sec)": [2.298217535018921, 7.478943586349487, 1592.7058815956116, 15164.204834222794]})

    adelphi_df = pd.read_csv('data/adelphi_fpt.csv')

    for floor in range(round(adelphi_df["# Floors"].max()) + 1, num_floors + 1):
        print(f"Floor {floor}/{num_floors}")
        tree = adelphi_tree(num_floors=floor)

        for robots in range(1, num_robots + 1):
            print(f"Robots {robots}/{num_robots}")
            start = time()
            traversal = fpt_compute_traversal(tree, robots, heuristics_on=True, backtrack=True)
            end = time()

            adelphi_df.loc[len(adelphi_df)] = [floor, tree.size(), robots, len(traversal), end - start, (end - start) / 3600]
            print()
            print(f"Num robots = {robots}: ")
            print([floor, tree.size(), robots, len(traversal), end - start])
            print()

        # Save progress...
        adelphi_df.to_csv('data/adelphi_fpt.csv', index=False)  # Use index=False to avoid saving row indices

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    max_floor_df = adelphi_df[adelphi_df['# Floors'] == adelphi_df['# Floors'].max()].copy()
    ax1.plot(max_floor_df["# Robots"], max_floor_df["Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Robots", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    ax2.plot(max_floor_df["# Robots"], max_floor_df["Computation Time (hours)"], label="Computation Time (hours)",
             color="red")
    ax2.set_ylabel("Computation Time (hours)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    sns.lineplot(data=adelphi_df, x='# Vertices', y='Traversal Time', hue='# Robots')
    plt.show()

    adelphi_df.to_csv('data/adelphi_fpt.csv', index=False)  # Use index=False to avoid saving row indices


def compare_fpt_cocta(fpt_df, tree_generator):
    cocta_df = pd.DataFrame(columns=fpt_df.columns)

    for i in range(len(fpt_df)):
        entry = fpt_df.iloc[i]
        tree = tree_generator(num_floors=round(entry["# Floors"]))

        start = time()
        traversal = cocta_compute_traversal(tree, entry["# Robots"])
        end = time()

        cocta_df.loc[len(cocta_df)] = [entry["# Floors"], tree.size(), entry["# Robots"], len(traversal), end - start, (end - start) / 3600]

    fpt_df["% Saved Time"] = (cocta_df["Traversal Time"] / fpt_df["Traversal Time"] - 1) * 100

    sns.lineplot(data=fpt_df[fpt_df["# Floors"] > 1], x="# Floors", y="% Saved Time", hue="# Robots")
    plt.show()

