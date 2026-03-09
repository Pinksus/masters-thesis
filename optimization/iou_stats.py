import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class IoUStats:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.iou_values = None

    # -----------------------------
    # Load data
    # -----------------------------
    def load(self):
        self.iou_values = np.loadtxt(self.file_path)
        print(f"Loaded {len(self.iou_values)} IoU records.")

    # -----------------------------
    # Basic statistics
    # -----------------------------
    def print_basic_stats(self):
        iou = self.iou_values

        num_records = len(iou)
        mean_iou = np.mean(iou)
        median_iou = np.median(iou)
        std_iou = np.std(iou)
        min_iou = np.min(iou)
        max_iou = np.max(iou)
        q1 = np.percentile(iou, 25)
        q3 = np.percentile(iou, 75)

        print("\n=== IoU Statistics ===")
        print(f"Number of records: {num_records}")
        print(f"Mean: {mean_iou:.3f}")
        print(f"Median: {median_iou:.3f}")
        print(f"Std Dev: {std_iou:.3f}")
        print(f"Min: {min_iou:.3f}")
        print(f"Max: {max_iou:.3f}")
        print(f"Q1: {q1:.3f}, Q3: {q3:.3f}")

    # -----------------------------
    # IoU > threshold
    # -----------------------------
    def print_threshold_stats(self, threshold=0.5):
        iou = self.iou_values
        num_records = len(iou)

        count_above = np.sum(iou > threshold)
        percentage = (count_above / num_records) * 100

        print(f"\n=== IoU > {threshold} Statistics ===")
        print(f"IoU > {threshold}: {count_above} records")
        print(f"Percentage: {percentage:.2f}%")

    # -----------------------------
    # Frequency table
    # -----------------------------
    def print_frequency_table(self, step=0.05):
        intervals = np.arange(0, 1.0 + step, step)
        hist, bin_edges = np.histogram(self.iou_values, bins=intervals)

        print("\n=== IoU Frequency Table ===")
        for i in range(len(hist)):
            print(
                f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} : {hist[i]} occurrences"
            )

        return hist, bin_edges

    # -----------------------------
    # Plots
    # -----------------------------
    def plot_all(self):
        iou = self.iou_values
        num_records = len(iou)

        # Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(iou, bins=10, edgecolor='black', color='skyblue')
        plt.title("Histogram of IoU Values")
        plt.xlabel("IoU")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.7)
        plt.show()

        # CDF
        plt.figure(figsize=(8, 5))
        sorted_iou = np.sort(iou)
        cdf = np.arange(1, num_records + 1) / num_records
        plt.plot(sorted_iou, cdf, marker='o', linestyle='-', color='orange')
        plt.title("Cumulative Distribution of IoU Values")
        plt.xlabel("IoU")
        plt.ylabel("CDF")
        plt.grid(True)
        plt.show()

        # Boxplot
        plt.figure(figsize=(6, 5))
        plt.boxplot(
            iou,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='lightgreen'),
        )
        plt.title("Boxplot of IoU Values")
        plt.ylabel("IoU")
        plt.grid(axis='y', alpha=0.7)
        plt.show()

        # Violin
        plt.figure(figsize=(6, 5))
        sns.violinplot(y=iou, color='lightcoral')
        plt.title("Violin Plot of IoU Values")
        plt.ylabel("IoU")
        plt.grid(axis='y', alpha=0.7)
        plt.show()

        # Bar chart
        hist, bin_edges = np.histogram(iou, bins=np.arange(0, 1.05, 0.05))

        plt.figure(figsize=(10, 5))
        plt.bar(
            bin_edges[:-1],
            hist,
            width=0.045,
            align='edge',
            edgecolor='black',
            color='lightblue',
        )
        plt.xticks(bin_edges, rotation=45)
        plt.title("IoU Frequency by 0.05 Intervals")
        plt.xlabel("IoU Interval")
        plt.ylabel("Count")
        plt.grid(axis='y', alpha=0.7)
        plt.show()