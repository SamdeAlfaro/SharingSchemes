import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import seaborn as sns
import os

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

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

        filename = f"results/simplegraphs/graph_t{threshold}_n{num_shares}_s{secret_size}.png"
        plt.savefig(filename)
        plt.close()

def plot_stacked_histogram(df, x_col, title, filename):
    x_labels = df[x_col].astype(str)
    split = df['Split Time (ms)']
    verify = df['Verify Time (ms)']
    reconstruct = df['Reconstruct Time (ms)']

    x = np.arange(len(x_labels))
    bar_width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, split, bar_width, label="Split", color='#4dbd6b')
    ax.bar(x, verify, bar_width, bottom=split, label="Verify", color='#8f60cc')
    ax.bar(x, reconstruct, bar_width, bottom=split + verify, label="Reconstruct", color='#f24b7d')

    ax.set_xlabel(x_col)
    ax.set_ylabel('Average Time (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def generate_per_scheme_graphs(df):
    schemes = df['Scheme'].unique()

    for scheme in schemes:
        scheme_df = df[df['Scheme'] == scheme]

        # 1. Vary Secret Size (fixed threshold and num shares)
        fixed_params = scheme_df.groupby(['Threshold', 'Num Shares']).size().reset_index()
        for _, row in fixed_params.iterrows():
            threshold, num_shares = row['Threshold'], row['Num Shares']
            subset = scheme_df[(scheme_df['Threshold'] == threshold) & (scheme_df['Num Shares'] == num_shares)]
            if len(subset) > 1:
                plot_stacked_histogram(
                    subset,
                    x_col='Secret Size (bits)',
                    title=f'{scheme}: Vary Secret Size (t={threshold}, n={num_shares})',
                    filename=f'results/per_scheme/{scheme}_vary_secret_t{threshold}_n{num_shares}.png'
                )

        # 2. Vary Threshold (fixed secret size and num shares)
        fixed_params = scheme_df.groupby(['Secret Size (bits)', 'Num Shares']).size().reset_index()
        for _, row in fixed_params.iterrows():
            secret_size, num_shares = row['Secret Size (bits)'], row['Num Shares']
            subset = scheme_df[(scheme_df['Secret Size (bits)'] == secret_size) & (scheme_df['Num Shares'] == num_shares)]
            if len(subset) > 1:
                plot_stacked_histogram(
                    subset,
                    x_col='Threshold',
                    title=f'{scheme}: Vary Threshold (s={secret_size}, n={num_shares})',
                    filename=f'results/per_scheme/{scheme}_vary_threshold_s{secret_size}_n{num_shares}.png'
                )

        # 3. Vary Num Shares (fixed secret size, proportional threshold)
        fixed_secret_sizes = scheme_df['Secret Size (bits)'].unique()
        for secret_size in fixed_secret_sizes:
            subset = scheme_df[scheme_df['Secret Size (bits)'] == secret_size].copy()
            # Only consider cases where threshold/num_shares ratio is approximately constant
            subset['Ratio'] = subset['Threshold'] / subset['Num Shares']
            for ratio_val in subset['Ratio'].round(2).unique():
                ratio_subset = subset[subset['Ratio'].round(2) == ratio_val]
                if len(ratio_subset) > 1:
                    plot_stacked_histogram(
                        ratio_subset,
                        x_col='Num Shares',
                        title=f'{scheme}: Vary Num Shares (s={secret_size}, t/n≈{ratio_val})',
                        filename=f'results/per_scheme/{scheme}_vary_numshares_s{secret_size}_ratio{ratio_val}.png'
                    )

def generate_cross_scheme_comparisons(df):
    # Group by fixed secret size, threshold, num shares
    grouped = df.groupby(['Secret Size (bits)', 'Threshold', 'Num Shares'])

    for (secret_size, threshold, num_shares), group in grouped:
        if len(group['Scheme'].unique()) < 2:
            continue  # Only interesting if multiple schemes exist

        labels = group['Scheme']
        split_data = group['Split Time (ms)']
        verify_data = group['Verify Time (ms)']
        reconstruct_data = group['Reconstruct Time (ms)']

        x = np.arange(len(labels))
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x, split_data, bar_width, label="Split", color='#4dbd6b')
        ax.bar(x, verify_data, bar_width, bottom=split_data, label="Verify", color='#8f60cc')
        ax.bar(x, reconstruct_data, bar_width, bottom=split_data + verify_data, label="Reconstruct", color='#f24b7d')

        ax.set_xlabel('Scheme')
        ax.set_ylabel('Average Time (ms)')
        ax.set_title(f'Comparison: {num_shares} shares, threshold={threshold}, secret size={secret_size} bits')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--')
        plt.tight_layout()

        filename = f"results/cross_scheme/compare_t{threshold}_n{num_shares}_s{secret_size}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    csv_file = "results/benchmark_results.csv"
    df = load_data(csv_file)

    generate_per_scheme_graphs(df)
    generate_cross_scheme_comparisons(df)
    
    
    
    # These are the generic schemes
    # graph_from_csv(csv_file)