from matplotlib import pyplot as plt
import pandas as pd

def jaxonville_plot():
    jax_df = pd.DataFrame({
        "# Floors": [1, 2, 3, 4, 5],
        "# Vertices": [42, 82, 122, 162, 202],
        "# Traversal Time": [46, 98, 159, 211, 264],
        "# Computation Time (sec)": [670.23, 1733.03, 3428.89, 5611.87, 8239.31]
    })

    # Convert computation time from seconds to hours
    jax_df["# Computation Time (hours)"] = jax_df["# Computation Time (sec)"] / 3600

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Plot Traversal Time
    ax1.plot(jax_df["# Vertices"], jax_df["# Traversal Time"], label="Traversal Time", color="blue")
    ax1.set_xlabel("# Vertices", fontsize=20)
    ax1.set_ylabel("# Traversal Time", color="blue", fontsize=20)
    ax1.tick_params(axis='y', labelcolor="blue", labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)

    # Set up secondary y-axis for Computation Time
    ax2 = ax1.twinx()
    ax2.plot(jax_df["# Vertices"], jax_df["# Computation Time (hours)"], label="Computation Time (hours)", color="red")
    ax2.set_ylabel("# Computation Time (hours)", color="red", fontsize=20)
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
