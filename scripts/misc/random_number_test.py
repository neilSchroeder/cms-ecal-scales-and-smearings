import math
import time

import numba
import numpy as np
from numpy.random import PCG64, Generator
from scipy import stats

# Size of random numbers to generate
N = 1_000_000


def benchmark_function(func, *args):
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    return end_time - start_time, result


# 1. Standard numpy.random.normal
def numpy_normal(mean=0.0, std=1.0):
    return np.random.normal(loc=mean, scale=std, size=N)


# 2. Box-Muller transform implementation
@numba.jit(nopython=True)
def box_muller_numba(size, mean=0.0, std=1.0):
    result = np.empty(size)
    for i in range(0, size, 2):
        u1 = np.random.random()
        u2 = np.random.random()

        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2

        result[i] = r * math.cos(theta) * std + mean
        if i + 1 < size:
            result[i + 1] = r * math.sin(theta) * std + mean
    return result


# 3. PCG64 generator
def pcg64_normal(mean=0.0, std=1.0):
    rng = Generator(PCG64())
    return rng.normal(loc=mean, scale=std, size=N)


# 4. Ziggurat method via scipy
def scipy_normal(mean=0.0, std=1.0):
    return stats.norm.rvs(loc=mean, scale=std, size=N)


# 5. Vectorized Box-Muller
def box_muller_vectorized(mean=0.0, std=1.0):
    u1 = np.random.random(size=(N + 1) // 2)
    u2 = np.random.random(size=(N + 1) // 2)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2

    x = r * np.cos(theta) * std + mean
    y = r * np.sin(theta) * std + mean

    return np.concatenate([x, y])[:N]


if __name__ == "__main__":
    # Run benchmarks
    # Set test parameters
    mean = 1.0  # Example mean
    std = 0.02  # Example standard deviation

    methods = [
        ("NumPy normal", numpy_normal, mean, std),
        ("Box-Muller (Numba)", box_muller_numba, N, mean, std),
        ("PCG64 Generator", pcg64_normal, mean, std),
        ("SciPy norm.rvs", scipy_normal, mean, std),
        ("Box-Muller (Vectorized)", box_muller_vectorized, mean, std),
    ]

    results = {}
    samples = {}

    for method in methods:
        name = method[0]
        func = method[1]
        args = method[2:] if len(method) > 2 else ()

        print(f"\nRunning {name}...")
        time_taken, numbers = benchmark_function(func, *args)

        results[name] = time_taken
        samples[name] = numbers

        print(f"Time taken: {time_taken:.4f} seconds")
        print(f"Mean: {np.mean(numbers):.6f}")
        print(f"Std: {np.std(numbers):.6f}")

    # Print summary
    print("\nSummary (sorted by speed):")
    for name, time_taken in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: {time_taken:.4f} seconds")
