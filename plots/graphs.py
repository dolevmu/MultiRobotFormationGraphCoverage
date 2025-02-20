from time import time
from tqdm import tqdm

from matplotlib import pyplot as plt
import concurrent.futures
import pandas as pd
import seaborn as sns

from trees.cocta import cocta_compute_traversal
from trees.signature import enumerate_signatures
from trees.table import fpt_compute_traversal
from trees.tree import adelphi_tree, jaxsonville_tree, random_building_tree, MAX_HALL_LENGTH, MIN_HALL_LENGTH, \
    MAX_HALLS_PER_FLOOR, ROOM_DENSITY, NUM_FLOORS, add_floors_to_tree, increase_room_density, stretch_halls


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
    adelphi_df = pd.read_csv('data/adelphi_fpt.csv')

    if adelphi_df.shape[0] == 0:
        start_floor = 0
    else:
        start_floor = round(adelphi_df["# Floors"].max()) + 1

    for floor in range(start_floor, num_floors + 1):
        print(f"Floor {floor}/{num_floors}")
        tree = adelphi_tree(num_floors=floor)

        for robots in range(1, num_robots + 1):
            print(f"Robots {robots}/{num_robots}")
            start = time()
            traversal = fpt_compute_traversal(tree, robots, heuristics_on=True, backtrack=True, max_sig_length=max_sig_length)
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


def plot_computation_time_graph(fpt_df):
    computation_time = "Computation Time (hours)"
    number_of_nodes = "# Vertices"

    sns.lineplot(data=fpt_df[fpt_df["# Floors"] > 1],
                 x=number_of_nodes,
                 y=computation_time,
                 hue="# Robots")
    plt.show()

def adelphi_avg_num_signatures():
    max_sig_length = 8
    num_robots = 4
    robots_to_sig_lengths = {1: range(1, max_sig_length),
                             2: range(1, max_sig_length+1),
                             3: range(1, max_sig_length+4),
                             4: range(1, max_sig_length+3)}

    # df = pd.DataFrame(columns=["# Robots", "Signature Max Length", "Avg. # Signatures"])
    df = pd.read_csv("data/adelphi_pacman_avg_sigs.csv")

    floor = 4
    tree = adelphi_tree(num_floors=floor)

    for robots in range(1, num_robots + 1):
    # for robots in robots_to_sig_lengths:
        print(f"Robots {robots}/{num_robots}")
        for sig_length in robots_to_sig_lengths[robots]:
            print(f"Max Sig Length={sig_length}")

            if sum((df["# Robots"] == robots) &
                   (df["Signature Max Length"] == sig_length)) > 0:
                continue # Already computed

            avg_num_signatures = 0
            for vertex in tree.nodes:
                counter = 0
                signatures_iterator = enumerate_signatures(vertex, tree, robots, max_sig_length=sig_length)
                for _ in tqdm(signatures_iterator, desc=f"Vertex={vertex: >4}"):
                    counter += 1
                avg_num_signatures += counter
            avg_num_signatures /= tree.size()
            df.loc[len(df)] = [robots, sig_length, avg_num_signatures]

    df.to_csv('data/adelphi_pacman_avg_sigs.csv', index=False)

    g = sns.barplot(data=df, x='# Robots', y="Avg. # Signatures", hue="Signature Max Length")
    g.set_yscale("log")
    plt.show()


def floor_random_graph_plots(num_samples: int = 100,
                             load: bool = True,
                             suffix: str = ''):
    num_robots = 3
    num_floors = NUM_FLOORS
    max_floors_to_add = 10 - num_floors
    max_num_floors = num_floors + max_floors_to_add
    room_density = ROOM_DENSITY
    num_halls_per_floor = MAX_HALLS_PER_FLOOR
    min_hall_length = MIN_HALL_LENGTH
    max_hall_length = MAX_HALL_LENGTH

    trees = [random_building_tree() for _ in range(num_samples)]

    if load:
        floor_df = pd.read_csv(f'data/floor_df{suffix}.csv')

    else:
        floor_df = pd.DataFrame(columns=["# Vertices", "# Floors", "# Halls", "# Rooms", "# Halls per floor", "Min. hall length", "Max. hall length", "Room Density", "Traversal Time", "% Saved Time", "Computation Time (sec)", "Computation Time (min)"])

        for floors_to_add in range(max_floors_to_add):
            for sample in range(num_samples):
                print(f"Floor {num_floors}/{max_num_floors}")

                tree = trees[sample]
                num_vertices = tree.size()
                num_halls = sum(v.startswith("Branch") for v in tree.nodes) + num_floors
                num_rooms = sum(v.startswith("Room") for v in tree.nodes)

                pacman_time = time()
                try:
                    pacman_traversal = fpt_compute_traversal(tree, num_robots, heuristics_on=True, backtrack=True,
                                                             max_sig_length=9)
                except:
                    print("Failed")
                    continue
                pacman_time = time() - pacman_time

                # cocta_time = time()
                try:
                    cocta_traversal = cocta_compute_traversal(tree, num_robots)
                except:
                    print("Failed")
                    continue
                # cocta_time = time() - cocta_time

                saved_time = (len(cocta_traversal) - len(pacman_traversal)) / len(cocta_traversal) * 100

                floor_df.loc[len(floor_df)] = [num_vertices, num_floors, num_halls, num_rooms, num_halls_per_floor,
                                             min_hall_length, max_hall_length, room_density,
                                             len(pacman_traversal), saved_time, pacman_time, pacman_time / 60]
                # Update tree for next iteration
                trees[sample] = add_floors_to_tree(tree, 1, num_floors=num_floors)
            num_floors += 1 # Update number of floors

            # Save progress...
            floor_df.to_csv(f'data/floor_df{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    # ax1.plot(floor_df["# Floors"], floor_df.groupby('# Floors')["Traversal Time"].transform('mean'), label="Traversal Time", color="blue")
    sns.lineplot(
        data=floor_df, x="# Floors", y="Traversal Time",
        ax=ax1, color="blue", label="Traversal Time",
        estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    ax1.set_xlabel("# Floors", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    # ax2.plot(floor_df["# Floors"], floor_df.groupby('# Floors')["% Saved Time"].transform('mean'), label="% Saved Time", color="red")
    sns.lineplot(
        data=floor_df, x="# Floors", y="% Saved Time",
        ax=ax1, color="red", label="% Saved Time",
        estimator="mean", errorbar="ci", err_kws={"alpha": 0.2}
    )
    ax2.set_ylabel("% Saved Time", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    floor_df.to_csv('data/floor_df.csv', index=False)  # Use index=False to avoid saving row indices


def density_random_graph_plots(num_samples: int = 100,
                               load: bool = True,
                               suffix: str = ''):
    num_floors = NUM_FLOORS
    num_robots = 3

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


def process_sample(tree, num_robots,
                   hall_length, max_hall_length,
                   room_density, num_floors,
                   num_halls_per_floor, min_hall_length):
    print(f"Hall Length {hall_length}/{max_hall_length}")

    num_vertices = tree.size()
    num_halls = sum(v.startswith("Branch") for v in tree.nodes) + num_floors
    num_rooms = sum(v.startswith("Room") for v in tree.nodes)

    pacman_time = time()
    try:
        pacman_traversal = fpt_compute_traversal(tree, num_robots, heuristics_on=True, backtrack=True, max_sig_length=9)
    except:
        print("Failed")
        pacman_traversal = []

    pacman_time = time() - pacman_time

    try:
        cocta_traversal = cocta_compute_traversal(tree, num_robots)
    except:
        print("Failed")
        return None

    saved_time = (len(cocta_traversal) - len(pacman_traversal)) / len(cocta_traversal) * 100 if cocta_traversal else 0

    result = [num_vertices, num_floors, num_halls, num_rooms, num_halls_per_floor,
              min_hall_length, hall_length, room_density,
              len(pacman_traversal), saved_time, pacman_time, pacman_time / 60]

    return result

def hall_random_graph_plots(num_samples: int = 100,
                            load: bool = True,
                            suffix: str = ''):
    num_robots = 3
    num_floors = 2
    room_density = 0.0
    max_halls_per_floor = 4
    min_hall_length = 3
    max_hall_length = 8

    trees = [random_building_tree() for _ in range(num_samples)]

    if load:
        floor_df = pd.read_csv(f'data/hall_df{suffix}.csv')

    else:
        floor_df = pd.DataFrame(columns=["# Vertices", "# Floors", "# Halls", "# Rooms", "# Halls per floor", "Min. hall length", "Max. hall length", "Room Density", "Traversal Time", "% Saved Time", "Computation Time (sec)", "Computation Time (min)"])

        for hall_length in range(min_hall_length, max_hall_length+1):
            parallel = False
            if parallel:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = {executor.submit(process_sample, trees[sample], num_robots,
                                               hall_length, max_hall_length,
                                               room_density, num_floors,
                                               max_halls_per_floor, min_hall_length): sample
                               for sample in range(num_samples)}
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        if result:
                            floor_df.loc[len(floor_df)] = result
                for sample in range(num_samples):
                    # Update tree for next iteration
                    trees[sample] = stretch_halls(trees[sample], hall_length_to_add=1, room_density=room_density)
            else:
                for sample in range(num_samples):
                    result = process_sample(trees[sample], num_robots,
                                            hall_length, max_hall_length,
                                            room_density, num_floors,
                                            max_halls_per_floor, min_hall_length)
                    if result:
                        floor_df.loc[len(floor_df)] = result
                    trees[sample] = stretch_halls(trees[sample], hall_length_to_add=1, room_density=room_density)

            # Save progress...
            floor_df.to_csv(f'data/hall_df{suffix}.csv', index=False)  # Use index=False to avoid saving row indices

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    ax1.plot(floor_df["Max. hall length"], floor_df.groupby("Max. hall length")["Traversal Time"].transform('mean'), label="Traversal Time", color="blue")
    ax1.set_xlabel("Max. hall length", fontsize=20)
    ax1.set_ylabel("Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    ax2.plot(floor_df["Max. hall length"], floor_df.groupby("Max. hall length")["% Saved Time"].transform('mean'), label="% Saved Time", color="red")
    ax2.set_ylabel("% Saved Time", color="red", fontsize=20)
    ax2.tick_params(axis='y', labelcolor="red", labelsize=18)

    fig.tight_layout()  # Adjust layout for clarity
    plt.show()

    floor_df.to_csv('data/hall_df.csv', index=False)  # Use index=False to avoid saving row indices

