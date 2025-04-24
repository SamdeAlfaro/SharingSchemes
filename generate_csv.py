from feldmansecretsharing import FeldmanVerifiableSecretSharing
from asmuthbloomsharing import AsmuthBloom
from blakleysecretsharing import BlakleySecretSharing
from pedersenscheme import PedersenSecretSharing
from shamirsecretsharing import ShamirSecretSharing

import datetime
import matplotlib.pyplot as plt
import numpy as np
import secrets
import csv

# ________________________________Defining Benmarking for Each Scheme__________________________________

def benchmark_shamir(secret, threshold, num_shares, iter):
    # Benhmark Shamir
    sss = ShamirSecretSharing(threshold, num_shares)
    split_times = []
    verify_times = []
    reconstruct_times = []

    for _ in range(iter):
        start_split = datetime.datetime.now()
        shares = sss.split(secret)
        split_times.append((datetime.datetime.now() - start_split).total_seconds() * 1000)
        
        start_recon = datetime.datetime.now()
        recovered = sss.reconstruct(shares[:threshold])
        reconstruct_times.append((datetime.datetime.now() - start_recon).total_seconds() * 1000)
    
    shamir_splittime = sum(split_times) / iter
    shamir_reconstructtime = sum(reconstruct_times) / iter
    return [shamir_splittime, 0, shamir_reconstructtime] # Use 0 as a placeholder for no verification step

def benchmark_pedersen(secret, threshold, num_shares, iter):
    pss = PedersenSecretSharing(threshold, num_shares)
    pedersen_split_times = []
    pedersen_verify_times = []
    pedersen_reconstruct_times = []

    for _ in range(iter):
        start_split = datetime.datetime.now()
        shares, commitments = pss.split(secret)
        pedersen_split_times.append((datetime.datetime.now() - start_split).total_seconds() * 1000)

        start_verify = datetime.datetime.now()
        for x, s, r in shares:
            pss.verify_share(x, s, r, commitments)
        pedersen_verify_times.append((datetime.datetime.now() - start_verify).total_seconds() * 1000)

        start_recon = datetime.datetime.now()
        recovered = pss.reconstruct([(x, s) for x, s, _ in shares[:threshold]])
        pedersen_reconstruct_times.append((datetime.datetime.now() - start_recon).total_seconds() * 1000)
    
    pedersen_splittime = sum(pedersen_split_times) / iter
    pedersen_verifytime = sum(pedersen_verify_times) / iter
    pedersen_reconstructtime = sum(pedersen_reconstruct_times) / iter
    return [pedersen_splittime, pedersen_verifytime, pedersen_reconstructtime]

def benchmark_feldman(secret, threshold, num_shares, iter):
    fvs = FeldmanVerifiableSecretSharing(threshold, num_shares)
    feldman_split_times = []
    feldman_verify_times = []
    feldman_reconstruct_times = []

    for _ in range(iter):
        start_split = datetime.datetime.now()
        shares, commitments = fvs.split(secret)
        feldman_split_times.append((datetime.datetime.now() - start_split).total_seconds() * 1000)

        start_verify = datetime.datetime.now()
        for x, s in shares:
            fvs.verify_share(x, s, commitments)
        feldman_verify_times.append((datetime.datetime.now() - start_verify).total_seconds() * 1000)

        start_recon = datetime.datetime.now()
        recovered = fvs.reconstruct([(x, s) for x, s in shares[:threshold]])
        feldman_reconstruct_times.append((datetime.datetime.now() - start_recon).total_seconds() * 1000)

    feldman_splittime = sum(feldman_split_times) / iter
    feldman_verifytime = sum(feldman_verify_times) / iter
    feldman_reconstructtime = sum(feldman_reconstruct_times) / iter
    return [feldman_splittime, feldman_verifytime, feldman_reconstructtime]

def benchmark_blakley(secret, threshold, num_shares, iter):
    bss = BlakleySecretSharing(threshold, num_shares)
    blakley_split_times = []
    blakley_reconstruct_times = []

    for _ in range(iter):
        start_split = datetime.datetime.now()
        shares = bss.split(secret)
        blakley_split_times.append((datetime.datetime.now() - start_split).total_seconds() * 1000)

        start_recon = datetime.datetime.now()
        recovered = bss.reconstruct([(x, s) for x, s in shares[:threshold]])
        blakley_reconstruct_times.append((datetime.datetime.now() - start_recon).total_seconds() * 1000)

    blakley_splittime = sum(blakley_split_times) / iter
    blakley_reconstructtime = sum(blakley_reconstruct_times) / iter
    return [blakley_splittime, 0, blakley_reconstructtime]

def benchmark_ab(secret, threshold, num_shares, iter):
    ab = AsmuthBloom(threshold, num_shares)
    ab_split_times = []
    ab_reconstruct_times = []

    for _ in range(iter):
        start_split = datetime.datetime.now()
        shares = ab.split(secret)
        ab_split_times.append((datetime.datetime.now() - start_split).total_seconds() * 1000)

        start_recon = datetime.datetime.now()
        recovered = ab.reconstruct([(x, s) for x, s in shares[:threshold]])
        ab_reconstruct_times.append((datetime.datetime.now() - start_recon).total_seconds() * 1000)

    ab_splittime = sum(ab_split_times) / iter
    ab_reconstructtime = sum(ab_reconstruct_times) / iter

    return [ab_splittime, 0, ab_reconstructtime]

def benchmark(secret, threshold, num_shares, iter):
    # Benhmark Shamir
    shamir_results = benchmark_shamir(secret, threshold, num_shares, iter)

    # Benchmark Pedersen
    pedersen_results = benchmark_pedersen(secret, threshold, num_shares, iter)

    # Benchmark Feldman
    feldman_results = benchmark_feldman(secret, threshold, num_shares, iter)

    # Benchmark Blakley
    blakley_results = benchmark_blakley(secret, threshold, num_shares, iter)

    # Benchmark Asmuth-Bloom
    ab_results = benchmark_ab(secret, threshold, num_shares, iter)

    return shamir_results, pedersen_results, feldman_results, blakley_results, ab_results




# ________________________________Graph Results__________________________________

# This is
def graph_results(threshold, num_shares, secret_size, shamir_results, pedersen_results, feldman_results, blakley_results, ab_results):
    # Combine results for graphing
    split_data = [shamir_results[0], pedersen_results[0], feldman_results[0], blakley_results[0], ab_results[0]]
    verify_data = [shamir_results[1], pedersen_results[1], feldman_results[1], blakley_results[1], ab_results[1]]
    reconstruct_data = [shamir_results[2], pedersen_results[2], feldman_results[2], blakley_results[2], ab_results[2]]
    labels = ["Shamir", "Pedersen", "Feldman", "Blakley", "Asmuth-Bloom"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bar_width = 0.25

    split_color = '#4dbd6b'
    verify_color = '#8f60cc'
    reconstruct_color = '#f24b7d'

    ax.bar(x, split_data, bar_width, label="Split", color=split_color)
    ax.bar(x, verify_data, bar_width, bottom=split_data, label="Verify", color=verify_color)
    ax.bar(x, reconstruct_data, bar_width, bottom=np.array(split_data) + np.array(verify_data), label="Reconstruct", color=reconstruct_color)

    ax.set_xlabel('Secret Sharing Scheme')
    ax.set_ylabel('Average Time (milliseconds)')
    ax.set_title(f'Benchmarking Secret Sharing Schemes\nfor {num_shares} shares, threshold = {threshold}, and secret size (in bits) {secret_size}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()

    # Add a light grid
    ax.grid(axis='y', linestyle='--')

    plt.tight_layout()

    filename = f"results/benchmark_t{threshold}_n{num_shares}_s{secret_size}.png"
    # Save the figure
    plt.savefig(filename)

# ________________________________Benchmark!__________________________________


iterations = 1
thresholds = [[2, 5, 10], [5, 50, 100], [5, 500, 1000], [5, 5000, 10000]]
shares = 10
bit_lengths = [32, 64, 128, 256]
secrets_list = [secrets.randbits(bits) for bits in bit_lengths]

csv_filename = "results/benchmark_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Secret Size (bits)', 'Threshold', 'Num Shares', 'Scheme', 'Split Time (ms)', 'Verify Time (ms)', 'Reconstruct Time (ms)'])

    for i in range(4):  # over secret sizes
        secret = secrets_list[i]
        shares = 10 * (10 ** i)  # start with 10 and scale: 10, 100, 1000, 10000
        for j in range(3):  # thresholds within each shares group
            threshold = thresholds[i][j]
            print(f"Running benchmark for secret size {bit_lengths[i]} bits, threshold {threshold}, and {shares} shares ......")

            # Run benchmark
            shamir_results, pedersen_results, feldman_results, blakley_results, ab_results = benchmark(secret, threshold, shares, iterations)

            # Write CSV
            for label, split, verify, reconstruct in zip(
                ["Shamir", "Pedersen", "Feldman", "Blakley", "Asmuth-Bloom"],
                [shamir_results[0], pedersen_results[0], feldman_results[0], blakley_results[0], ab_results[0]],
                [shamir_results[1], pedersen_results[1], feldman_results[1], blakley_results[1], ab_results[1]],
                [shamir_results[2], pedersen_results[2], feldman_results[2], blakley_results[2], ab_results[2]]
            ):
                writer.writerow([bit_lengths[i], threshold, shares, label, split, verify, reconstruct])

            print(f"Completed benchmark for secret size {bit_lengths[i]} bits, threshold {threshold}, and {shares} shares")
