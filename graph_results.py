import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def graph_from_csv(csv_file):
    grouped_data = defaultdict(lambda: {'labels': [], 'split': [], 'verify': [], 'reconstruct': []})

    with open(csv_file, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (int(row['Secret Size (bits)']), int(row['Threshold']), int(row['Num Shares']))
            grouped_data[key]['labels'].append(row['Scheme'])
            grouped_data[key]['split'].append(float(row['Split Time (ms)']))
            grouped_data[key]['verify'].append(float(row['Verify Time (ms)']))
            grouped_data[key]['reconstruct'].append(float(row['Reconstruct Time (ms)']))

    for (secret_size, threshold, num_shares), data in grouped_data.items():
        labels = data['labels']
        split_data = data['split']
        verify_data = data['verify']
        reconstruct_data = data['reconstruct']

        x = np.arange(len(labels))
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x, split_data, bar_width, label="Split", color='#4dbd6b')
        ax.bar(x, verify_data, bar_width, bottom=split_data, label="Verify", color='#8f60cc')
        ax.bar(x, reconstruct_data, bar_width, bottom=np.array(split_data) + np.array(verify_data), label="Reconstruct", color='#f24b7d')

        ax.set_xlabel('Secret Sharing Scheme')
        ax.set_ylabel('Average Time (ms)')
        ax.set_title(f'Benchmark: {num_shares} shares, threshold={threshold}, secret size={secret_size} bits')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--')
        plt.tight_layout()

        filename = f"results/graph_t{threshold}_n{num_shares}_s{secret_size}.png"
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    csv_file = "results/benchmark_results.csv"
    graph_from_csv(csv_file)