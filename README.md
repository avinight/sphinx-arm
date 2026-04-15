# Sphinx: A Succinct Perfect Hash Index for x86 & ARM64

<p align="left">
Sphinx is a state-of-the-art succinct perfect hash table engineered for high performance on modern CPUs. Its innovative encoding leverages rank and select primitives alongside auxiliary metadata to enable near-instantaneous hash table slot decoding.

Sphinx is optimized for both **x86_64 (AVX2/BMI2)** and **ARM64 (NEON)**.
</p>


VLDB 2025 paper: https://www.vldb.org/pvldb/vol18/p4424-maghrebi.pdf

## Quickstart

To get started, simply clone the repository and navigate into the project directory:

The easiest way to build Sphinx and reproduce the paper's results is using our integrated automation script:

```bash
# Clone and enter the repo
git clone [repo-url]
cd sphinx

# Run the full pipeline (Stages 1-5)
# Stage 5 generates the final visualizations
./benchmark/reproduce.sh --stage 5
```

## Reproducablity
Benchmark outputs will be generated in the `build/benchmark/` directory. For complete instructions on reproducing the results described in the paper, see [reproducibility.md](benchmark/reproducibility.md).

## Compiling and Testing

After cloning the repository along with its submodules, follow these steps to build the project using CMake:

1. Create a build directory and navigate into it:
   ```bash
   cmake -B build . 
   cmake --build build --parallel $(sysctl -n hw.ncpu || nproc)
   ```

2. To run tests, make sure that `ENABLE_MT` is defined in your configuration. Then, execute:
   ```bash
   ctest --test-dir build
   ```

## License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for full details.
