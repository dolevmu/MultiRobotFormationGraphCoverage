from time import time
from tqdm import tqdm

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from trees.table import fpt_compute_traversal
from trees.tree import adelphi_tree


def jaxonville_plot():
    jax_df = pd.DataFrame({
        "# Floors": [1, 2, 3, 4, 5],
        "# Vertices": [42, 82, 122, 162, 202],
        "Traversal Time": [46, 98, 159, 211, 264],
        "Computation Time (sec)": [670.23, 1733.03, 3428.89, 5611.87, 8239.31]
    })

    # Convert computation time from seconds to hours
    jax_df["Computation Time (hours)"] = jax_df["Computation Time (sec)"] / 3600

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    ax1.plot(jax_df["# Vertices"], jax_df["Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Vertices", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    ax2.plot(jax_df["# Vertices"], jax_df["Computation Time (hours)"], label="Computation Time (hours)", color="red")
    ax2.set_ylabel("Computation Time (hours)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    # Set up secondary x-axis for # Floors
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())  # Match the x-axis limits with the primary x-axis
    ax3.set_xticks(jax_df["# Vertices"])  # Set the same x-axis tick positions as # Vertices
    ax3.set_xticklabels(jax_df["# Floors"], fontsize=18)  # Label ticks as # Floors
    ax3.set_xlabel("# Floors", fontsize=20)

    # Optional title
    # plt.title("Jaxonville Hotel: Traversal and Computation Time vs Graph Size", fontsize=18)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()


def adelphi_plot(num_floors: int):  # Adelphi Hotel, Melbourne

    # adelphi_df = pd.DataFrame(data={"# Floors": [],
    #                                 "# Vertices": [],
    #                                 "Traversal Time": [],
    #                                 "Computation Time (sec)": [],
    #                                 "Heuristics": []})

    adelphi_df = pd.DataFrame(data={"# Floors": [1, 1, 2, 2, 3, 3],
                                    "# Vertices": [2, 2, 19, 19, 36, 36],
                                    "Traversal Time": [2, 2,
                                                       19, 19,
                                                       45, 56],
                                    "Computation Time (sec)": [0.04491138458251953, 0.041311025619506836,
                                                               457.55689001083374, 110.27886056900024,
                                                               1631.712432384491, 918.9196190834045],
                                    "Heuristics": ["Off", "On", "Off", "On", "Off", "On"]})


    # adelphi_df = pd.DataFrame(data={"# Floors": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    #                                 "# Vertices": [2, 2, 19, 19, 36, 36, 53, 53, 70, 70],
    #                                 "Traversal Time": [2, 2,
    #                                                    19, 19,
    #                                                    45, 56,
    #                                                    71, 83,
    #                                                    97, 110],
    #                                 "Computation Time (sec)": [0.0759727954864502, 0.04541015625,
    #                                                            244.18337988853455, 83.10970425605774,
    #                                                            5384.65568614006, 45812.05362486839,
    #                                                            3185.543482065201, 2834.186688184738,
    #                                                            5142.664044380188, 6190.431360960007],
    #                                 "Heuristics": ["Off", "On",
    #                                                "Off", "On",
    #                                                "Off", "On",
    #                                                "Off", "On",
    #                                                "Off", "On"]})


    computed = len(adelphi_df) // 2
    for floor in range(computed + 1, num_floors + 1):
        print(f"Floor {floor}/{num_floors}")
        tree = adelphi_tree(num_floors=floor)

        precise_start = time()
        precise_traversal = fpt_compute_traversal(tree, 2, heuristics_on=False)
        precise_end = time()

        heuristic_start = time()
        heuristic_traversal = fpt_compute_traversal(tree, 2, heuristics_on=True)
        heuristic_end = time()

        print(f"Precise: {precise_end-precise_start}")
        print(f"Heuristic: {heuristic_end-heuristic_start}")

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

def adelphi_robots_plot(num_robots: int):
    tree = adelphi_tree(num_floors=5)

    adelphi_df = pd.DataFrame(data={"# Floors": [5, 5, 5],
                                    "# Vertices": [70, 70, 70],
                                    "# Robots": [1, 2, 3],
                                    "Traversal Time": [125, 110, 91],
                                    "Computation Time (sec)": [2.329224, 4.141321, 534.187461]})
    computed = len(adelphi_df)
    total = num_robots - computed
    for robots in range(computed + 1, num_robots + 1):
        print(f"Robots {robots}/{num_robots}")
        start = time()
        traversal = fpt_compute_traversal(tree, robots, heuristics_on=True)
        end = time()

        adelphi_df.loc[len(adelphi_df)] = [5, tree.size(), robots, len(traversal), end - start]

    # Convert computation time from seconds to hours
    adelphi_df["Computation Time (hours)"] = adelphi_df["Computation Time (sec)"] / 3600

    print(adelphi_df)

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    ax1.plot(adelphi_df["# Robots"], adelphi_df["Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Robots", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    ax2.plot(adelphi_df["# Robots"], adelphi_df["Computation Time (hours)"], label="Computation Time (hours)",
             color="red")
    ax2.set_ylabel("Computation Time (hours)", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()



