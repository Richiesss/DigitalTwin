import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("pose_output_selected.csv")

# Filter valid tracking IDs
df = df.dropna(subset=["tracking_id"])
df["tracking_id"] = df["tracking_id"].astype(int)

# Remove rows where nose_y is 0 or NaN
df = df[(df["nose_y"].notna()) & (df["nose_y"] != 0)]

# Convert look_down to boolean
def to_bool(x):
    if isinstance(x, str):
        return x.lower() == "true"
    return bool(x)

df["look_down"] = df["look_down"].apply(to_bool)

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

# Create bar chart
plt.figure(figsize=(10, 6))

# Extract data for plotting
student_ids = stats_df["Tracking ID"]
concentration_rates = stats_df["Concentration Rate (%)"]

# Create x-axis positions (0, 1, 2, ...) for compacted display
x_positions = range(len(student_ids))

# Create bar chart with color gradient based on concentration
colors = plt.cm.RdYlGn(concentration_rates / 100)

bars = plt.bar(x_positions, concentration_rates, color=colors, edgecolor="black", linewidth=1.2)

# Add value labels on top of bars
for bar, rate in zip(bars, concentration_rates):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.xlabel("Student ID", fontsize=12, fontweight="bold")
plt.ylabel("Concentration Rate (%)", fontsize=12, fontweight="bold")
plt.title("Student Concentration Rates", fontsize=14, fontweight="bold")
plt.ylim(0, 105)
# Set x-ticks to show actual student IDs at compacted positions
plt.xticks(x_positions, student_ids)
plt.grid(True, axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("concentration_bar_chart.png", dpi=300)
print("\n棒グラフを保存しました: concentration_bar_chart.png")
