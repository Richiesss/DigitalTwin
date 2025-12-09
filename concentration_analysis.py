import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
df = pd.read_csv("pose_output_selected.csv")

# Filter valid tracking IDs
df = df.dropna(subset=["tracking_id"])
df["tracking_id"] = df["tracking_id"].astype(int)

# Filter out invalid coordinates (assuming (0,0) is invalid for nose)
# Check how many such rows exist
invalid_nose_count = df[(df["nose_x"] == 0) & (df["nose_y"] == 0)].shape[0]
print(f"Rows with nose at (0,0): {invalid_nose_count}")

# Remove rows where nose_y is 0 or NaN (as this affects the look_down calculation significantly)
df = df[(df["nose_y"].notna()) & (df["nose_y"] != 0)]

# Fix look_down column type
# Inspect unique values again to be sure
print("Unique look_down values before fix:", df["look_down"].unique())


# Robust conversion to boolean
# If it's string 'True'/'False', convert. If it's boolean, keep.
def to_bool(x):
    if isinstance(x, str):
        return x.lower() == "true"
    return bool(x)


df["look_down"] = df["look_down"].apply(to_bool)

print("Unique look_down values after fix:", df["look_down"].unique())

# Calculate Statistics
stats = []
ids = sorted(df["tracking_id"].unique())

for tid in ids:
    id_data = df[df["tracking_id"] == tid]
    total_frames = len(id_data)
    look_down_frames = id_data["look_down"].sum()

    # Concentration: Percentage of time NOT looking down
    concentration_rate = (
        100 * (1 - (look_down_frames / total_frames)) if total_frames > 0 else 0
    )

    stats.append(
        {
            "Tracking ID": tid,
            "Total Frames": total_frames,
            "Look Down Frames": look_down_frames,
            "Concentration Rate (%)": concentration_rate,
        }
    )

stats_df = pd.DataFrame(stats)
print("\nConcentration Statistics:")
print(stats_df)

# Plotting
plt.figure(figsize=(14, 8))

# Define colors
color_map = {True: "red", False: "green"}

# Plot each ID's timeline
# We will plot a point for every frame
# Y-axis: Tracking ID (compacted to sequential positions)
# X-axis: Frame Number

# Create a mapping from tracking ID to y-position (0, 1, 2, ...)
id_to_position = {tid: idx for idx, tid in enumerate(ids)}

for tid in ids:
    id_data = df[df["tracking_id"] == tid].sort_values("frame")

    # To make plotting faster and cleaner, we can plot segments
    # But scatter is easiest for variable data
    colors = id_data["look_down"].map(color_map)

    # Use the compacted position instead of the actual tracking ID
    y_position = id_to_position[tid]
    plt.scatter(
        id_data["frame"], [y_position] * len(id_data), c=colors, s=15, marker="|", alpha=0.8
    )

plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Student ID", fontsize=12)
plt.title(
    "Concentration Timeline: Looking Down (Red) vs. Concentrated (Green)", fontsize=14
)
# Set y-ticks to show actual tracking IDs at compacted positions
plt.yticks(range(len(ids)), ids)
plt.grid(True, axis="y", linestyle="--", alpha=0.5)

# Legend
red_patch = mpatches.Patch(color="red", label="Looking Down (Distracted)")
green_patch = mpatches.Patch(color="green", label="Concentrated")
plt.legend(handles=[green_patch, red_patch], loc="upper right")

plt.tight_layout()
plt.savefig("concentration_analysis_final.png")
