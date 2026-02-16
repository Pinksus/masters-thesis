import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load IoU data
# -----------------------------
file_path = "./files_whole_dataset/_speed_5-100_distance_50_bev_iou_moved.txt"  # change if needed
iou_values = np.loadtxt(file_path)

# -----------------------------
# Basic statistics
# -----------------------------
num_records = len(iou_values)
mean_iou = np.mean(iou_values)
median_iou = np.median(iou_values)
std_iou = np.std(iou_values)
min_iou = np.min(iou_values)
max_iou = np.max(iou_values)
q1 = np.percentile(iou_values, 25)
q3 = np.percentile(iou_values, 75)

print("=== IoU Statistics ===")
print(f"Number of records: {num_records}")
print(f"Mean: {mean_iou:.3f}")
print(f"Median: {median_iou:.3f}")
print(f"Std Dev: {std_iou:.3f}")
print(f"Min: {min_iou:.3f}")
print(f"Max: {max_iou:.3f}")
print(f"Q1: {q1:.3f}, Q3: {q3:.3f}")

# -----------------------------
# IoU > 0.5 statistics
# -----------------------------
threshold = 0.5
count_above_threshold = np.sum(iou_values > threshold)
percentage_above_threshold = (count_above_threshold / num_records) * 100

print("\n=== IoU > 0.5 Statistics ===")
print(f"IoU > {threshold}: {count_above_threshold} records")
print(f"Percentage: {percentage_above_threshold:.2f}%")


# -----------------------------
# Divide IoU into 0.05 intervals
# -----------------------------
intervals = np.arange(0, 1.05, 0.05)
hist, bin_edges = np.histogram(iou_values, bins=intervals)

print("\n=== IoU Frequency Table (0.05 bins) ===")
for i in range(len(hist)):
    print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} : {hist[i]} occurrences")

# -----------------------------
# Histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(iou_values, bins=10, edgecolor='black', color='skyblue')
plt.title("Histogram of IoU Values")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.7)
plt.show()

# -----------------------------
# Cumulative Distribution Function (CDF)
# -----------------------------
plt.figure(figsize=(8,5))
sorted_iou = np.sort(iou_values)
cdf = np.arange(1, num_records+1) / num_records
plt.plot(sorted_iou, cdf, marker='o', linestyle='-', color='orange')
plt.title("Cumulative Distribution of IoU Values")
plt.xlabel("IoU")
plt.ylabel("CDF")
plt.grid(True)
plt.show()

# -----------------------------
# Boxplot
# -----------------------------
plt.figure(figsize=(6,5))
plt.boxplot(iou_values, vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title("Boxplot of IoU Values")
plt.ylabel("IoU")
plt.grid(axis='y', alpha=0.7)
plt.show()

# -----------------------------
# Violin plot (distribution shape)
# -----------------------------
plt.figure(figsize=(6,5))
sns.violinplot(y=iou_values, color='lightcoral')
plt.title("Violin Plot of IoU Values")
plt.ylabel("IoU")
plt.grid(axis='y', alpha=0.7)
plt.show()

# -----------------------------
# Bar chart for 0.05 intervals
# -----------------------------
plt.figure(figsize=(10,5))
plt.bar(bin_edges[:-1], hist, width=0.045, align='edge', edgecolor='black', color='lightblue')
plt.xticks(bin_edges, rotation=45)
plt.title("IoU Frequency by 0.05 Intervals")
plt.xlabel("IoU Interval")
plt.ylabel("Count")
plt.grid(axis='y', alpha=0.7)
plt.show()
