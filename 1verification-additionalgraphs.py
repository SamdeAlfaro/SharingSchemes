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

def plot_grouped_stacked_histogram(df, group_col, x_col, title, filename):
    schemes = sorted(df[group_col].unique())
    metric_values = sorted(df[x_col].unique())

    bar_width = 0.15  # Narrower bars
    spacing = 0.5     # Extra space between different schemes

    total_bars = len(schemes) * len(metric_values)
    group_width = len(metric_values) * bar_width + spacing

    fig, ax = plt.subplots(figsize=(max(8, total_bars * 0.4), 6))

    colors = ['#4dbd6b', '#8f60cc', '#f24b7d']  # split, verify, reconstruct
    outlines = ['#3a8d52', '#6c48a1', '#c0395e']  # slightly darker outlines

    x_positions = []

    current_pos = 0
    for scheme in schemes:
        scheme_df = df[df['Scheme'] == scheme]
        for metric_val in metric_values:
            entry = scheme_df[scheme_df[x_col] == metric_val]
            if not entry.empty:
                split = entry['Split Time (ms)'].values[0]
                verify = entry['Verify Time (ms)'].values[0]
                reconstruct = entry['Reconstruct Time (ms)'].values[0]

                p1 = ax.bar(current_pos, split, bar_width, color=colors[0], edgecolor=outlines[0], label="Split" if current_pos == 0 else "", linewidth=0.7)
                p2 = ax.bar(current_pos, verify, bar_width, bottom=split, color=colors[1], edgecolor=outlines[1], label="Verify" if current_pos == 0 else "", linewidth=0.7)
                p3 = ax.bar(current_pos, reconstruct, bar_width, bottom=split + verify, color=colors[2], edgecolor=outlines[2], label="Reconstruct" if current_pos == 0 else "", linewidth=0.7)

                # Add tiny metric number above each bar
                total_height = split + verify + reconstruct
                ax.text(
                    current_pos,
                    total_height * 1.05,  # a bit above top
                    f"{metric_val}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=90
                )
            x_positions.append((current_pos, scheme))
            current_pos += bar_width
        current_pos += spacing  # extra space between schemes

    # Group labels (scheme names)
    scheme_positions = {}
    for xpos, scheme in x_positions:
        scheme_positions.setdefault(scheme, []).append(xpos)

    for scheme, positions in scheme_positions.items():
        center = np.mean(positions)
        ax.text(center, -0.2, scheme, ha='center', va='top', fontsize=10, fontweight='bold', transform=ax.get_xaxis_transform())

    ax.set_ylabel('Average Time (ms)')
    ax.set_title(title)
    ax.set_yscale('log')  # log scale
    ax.legend()
    ax.set_xticks([])  # Hide x-axis ticks (we manually labeled schemes)

    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

def plot_grouped_bars(df, x_col, title, filename):
    schemes = df['Scheme'].unique()
    spacing_within_scheme = 0.2
    spacing_between_schemes = 1.0
    bar_width = 0.2

    split_positions = []
    split_heights = []

    verify_positions = []
    verify_heights = []

    reconstruct_positions = []
    reconstruct_heights = []

    metric_annotations = []

    current_pos = 0

    for scheme in schemes:
        scheme_df = df[df['Scheme'] == scheme]
        scheme_df = scheme_df.sort_values(by=x_col)

        for _, row in scheme_df.iterrows():
            bars_in_group = 0
            group_positions = []

            # Always add split
            split_positions.append(current_pos)
            split_heights.append(row['Split Time (ms)'])
            group_positions.append(current_pos)
            current_pos += bar_width
            bars_in_group += 1

            # Only add verify if it exists
            if not (pd.isna(row['Verify Time (ms)']) or row['Verify Time (ms)'] == 0):
                verify_positions.append(current_pos)
                verify_heights.append(row['Verify Time (ms)'])
                group_positions.append(current_pos)
                current_pos += bar_width
                bars_in_group += 1

            # Always add reconstruct
            reconstruct_positions.append(current_pos)
            reconstruct_heights.append(row['Reconstruct Time (ms)'])
            group_positions.append(current_pos)
            current_pos += bar_width
            bars_in_group += 1

            group_center = np.mean(group_positions)
            metric_annotations.append((group_center, str(int(row[x_col]))))

            current_pos += spacing_within_scheme

        current_pos += spacing_between_schemes

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot bars
    ax.bar(split_positions, split_heights, width=bar_width, label='Split', color='#4dbd6b', edgecolor='#3aa957')
    ax.bar(verify_positions, verify_heights, width=bar_width, label='Verify', color='#8f60cc', edgecolor='#6f3eb0')
    ax.bar(reconstruct_positions, reconstruct_heights, width=bar_width, label='Reconstruct', color='#f24b7d', edgecolor='#c0305f')

    # ax.set_yscale('log')
    ax.set_ylabel('Average Time (ms)')
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove default x ticks
    ax.set_xticks([])

    # Add scheme labels centered under each scheme
    scheme_positions = []
    scheme_labels = []

    pos = 0
    for scheme in schemes:
        scheme_df = df[df['Scheme'] == scheme]
        num_metrics = len(scheme_df)

        # How many bars in total for the scheme?
        bars_per_metric = []
        for _, row in scheme_df.iterrows():
            bars = 2 if (pd.isna(row['Verify Time (ms)']) or row['Verify Time (ms)'] == 0) else 3
            bars_per_metric.append(bars)

        total_bars = sum(bars_per_metric)
        center_of_scheme = pos + (total_bars * bar_width + spacing_within_scheme * (num_metrics - 1)) / 2
        scheme_positions.append(center_of_scheme)
        scheme_labels.append(scheme)

        pos += total_bars * bar_width + spacing_within_scheme * (num_metrics - 1) + spacing_between_schemes

    ax.set_xticks(scheme_positions)
    ax.set_xticklabels(scheme_labels, rotation=0, fontsize=10)

    # for center, label in metric_annotations:
    #     ax.text(center, 0.5, label, ha='center', va='top', fontsize=8, rotation=0, clip_on=False)

    ax.legend()

    plt.tight_layout

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

    
def generate_compressed_graphs(df):
    # 1. Vary Secret Size (fixed threshold and num shares)
    fixed_params = df.groupby(['Threshold', 'Num Shares']).size().reset_index()
    for _, row in fixed_params.iterrows():
        threshold, num_shares = row['Threshold'], row['Num Shares']
        subset = df[(df['Threshold'] == threshold) & (df['Num Shares'] == num_shares)]
        if len(subset) > 1:
            plot_grouped_bars(
                subset,
                x_col='Secret Size (bits)',
                title=f'Vary Secret Size (t={threshold}, n={num_shares})',
                filename=f'results/verifytimes/vary_secret_t{threshold}_n{num_shares}.png'
            )

    # 2. Vary Threshold (fixed secret size and num shares)
    fixed_params = df.groupby(['Secret Size (bits)', 'Num Shares']).size().reset_index()
    for _, row in fixed_params.iterrows():
        secret_size, num_shares = row['Secret Size (bits)'], row['Num Shares']
        subset = df[(df['Secret Size (bits)'] == secret_size) & (df['Num Shares'] == num_shares)]
        if len(subset) > 1:
            plot_grouped_bars(
                subset,
                x_col='Threshold',
                title=f'Vary Threshold (s={secret_size}, n={num_shares})',
                filename=f'results/verifytimes/vary_threshold_s{secret_size}_n{num_shares}.png'
            )

    # 3. Vary Num Shares (fixed secret size, proportional threshold)
    fixed_secret_sizes = df['Secret Size (bits)'].unique()
    for secret_size in fixed_secret_sizes:
        subset = df[df['Secret Size (bits)'] == secret_size].copy()
        subset['Ratio'] = subset['Threshold'] / subset['Num Shares']
        for ratio_val in subset['Ratio'].round(2).unique():
            ratio_subset = subset[subset['Ratio'].round(2) == ratio_val]
            if len(ratio_subset) > 1:
                plot_grouped_bars(
                    ratio_subset,
                    x_col='Num Shares',
                    title=f'Vary Num Shares (s={secret_size}, t/n≈{ratio_val})',
                    filename=f'results/verifytimes/vary_numshares_s{secret_size}_ratio{ratio_val}.png'
                )

if __name__ == "__main__":
    csv_file = "results/verify_time.csv"
    df = load_data(csv_file)

    generate_compressed_graphs(df)
