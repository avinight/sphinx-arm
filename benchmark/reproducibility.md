# Reproducibility Guide

This document details the requirements, build instructions, environment setup, and procedures for generating figures to reproduce the experimental results presented in the paper.

## Requirements

Ensure that your system has the following software dependencies installed. Use **these versions and above**:

| Dependency     | Version Installed (or later) | Note |
|---------------|-----------------------------|---|
| **Seaborn**    | 0.12.2                      | Python plotting |
| **Matplotlib** | 3.8.0                       | Python plotting |
| **Pandas**     | 2.1.4                       | Data processing |
| **Numpy**      | 1.26.4                      | |
| **GCC**        | 11.4.0 (or Apple Clang)     | |
| **CMake**      | 3.22.1                      | |
| **Git**        | 2.34.1                      | |
| **Python**     | 3.11.7                      | |

> **Note:** For optimal performance, a machine with at least 16GB of RAM is recommended for standard benchmarks (64GB for full Linux scale-out). **ARM64 (Apple Silicon)** is fully supported via NEON SIMD optimizations.

---

## 1. Automated Workflow (Recommended)

The easiest way to reproduce the results is using the `reproduce.sh` script, which automates configuration, building, and execution across 5 stages.

### Standard Execution
```bash
./benchmark/reproduce.sh --stage [1-5]
```

### Pipeline Stages
| Stage | Description | Key Outputs |
|-------|-------------|-------------|
| **1** | SSD/Optane Baseline | `data-ssd/`, `data-tail/` |
| **2** | In-Memory & XDP Analysis | `data-memory/`, `data-memory-zipf/` |
| **3** | Advanced Workloads (Skew/MT) | `data-mt/`, `data-skew/` |
| **4** | Structural Properties | `data-extra-bits/`, `data-lf/` |
| **5** | Final Visualization | Generates all SVG figures in `benchmark/` |

### Cleanup Tools
Manage your workspace with the built-in cleanup flags:
- `--clean-build`: Removes `build/` artifacts and CMake cache.
- `--clean-data`: Removes collected CSVs and generated SVGs.
- `--deep-clean`: Full purge, including 16GB+ binary backing files.

---

## Building

For detailed build instructions, please refer to the [README.md](../README.md).

## Environment Setup

If you prefer to run benchmarks manually or on custom workloads, configure your environment by setting the appropriate flags in `./config/config.h`:

- **Memory Benchmarks:** Define `IN_MEMORY_FILE`
- **Total Memory Benchmarks:** Define `ENABLE_XDP`
- **Multi-threading Benchmarks:** Define `ENABLE_MT`
- **Skewed Workload Benchmarks:** Define `ENABLE_BP_FOR_READ`

> **Note:** The append-only log address should be updated to point to the correct mounted location. (For Optane benchmarks, ensure that the path includes the name `optane`.)

## Running Benchmarks and Generating Figures

1. **Execute Benchmarks:**

   Run the benchmark binaries (e.g., `./build/benchmark_*`) with the corresponding configuration. Upon execution, several data directories will be created:

| Directory       | Description                                                 | Figure Reference  |
|---------------|-------------------------------------------------|----------------|
| data-lf       | Load factor benchmark                         | Figure 14      |
| data-ssd      | Performance of Sphinx in SSD                 | Figure 10       |
| data-optane   | Performance of Sphinx in Optane              | Figure 10       |
| data-skew     | Performance of Sphinx under skew             | Figure 11 (Part A) |
| data-tail     | Measured tail latency                        | Figure 12       |
| data-memory   | Memory overhead & in-mem performance | Figures 7 & 8 & 9  |
| data-extra-bits | Reserve Bits                          | Figure 13      |
| data-mt       | Concurrency                   | Figure 11 (Part B) |

2. **Generate Figures:**

   After completing all benchmark runs, generate the paper figures by executing the following Python scripts:

   - **`mem_fpr.py`** – Generates Figure 7
   - **`main_exp_mem_ptr.py`** – Generates Figure 8 (run benchmark/benchmark_XDP.cpp beforehand)
   - **`main_exp_mem.py`** – Generates Figure 9
   - **`main_exp.py`** – Generates Figure 10
   - **`main_exp_zipf.py`** – Generates a figure for 50/50 update/query ycsb workload (set `constexpr bool MAIN_BENCHMARK_ZIPF = true;` in config/config.h, then run benchmark/benchmark_main.cpp)
   - **`combined.py`** – Generates Figure 11
   - **`tail.py`** – Generates Figure 12
   - **`extra_bits.py`** – Generates Figure 13
   - **`lf.py`** – Generates Figure 14
     

   Ensure that the data directories and configuration settings are correct before running these scripts.

