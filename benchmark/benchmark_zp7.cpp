#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "../bitset_wrapper/bitset_wrapper.h"
#include "../zp7/zp7.h"
#include "../zp7/simd_pdep_pext.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#if defined(__BMI2__)
#define HAS_BMI2 1
#else
#define HAS_BMI2 0
#endif
#else
#define HAS_BMI2 0
#endif

template <typename Fn>
static std::pair<double, uint64_t> run_bench_avg(const std::vector<uint64_t> &data,
                                                 const std::vector<uint64_t> &masks,
                                                 Fn fn,
                                                 size_t runs = 5) {
    double total_ms = 0.0;
    uint64_t checksum = 0;

    for (size_t run = 0; run < runs; ++run) {
        uint64_t run_checksum = 0;
        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < data.size(); ++i) {
            run_checksum ^= fn(data[i], masks[i]);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        total_ms += elapsed.count();

        if (run == 0) {
            checksum = run_checksum;
        }
    }

    return {total_ms / static_cast<double>(runs), checksum};
}

template <typename BatchFn>
static std::pair<double, uint64_t> run_bench_avg_batch(const std::vector<uint64_t> &data,
                                                       const std::vector<uint64_t> &masks,
                                                       BatchFn fn,
                                                       size_t runs = 5) {
    double total_ms = 0.0;
    uint64_t checksum = 0;
    std::vector<uint64_t> out(data.size(), 0);

    for (size_t run = 0; run < runs; ++run) {
        const auto start = std::chrono::high_resolution_clock::now();
        fn(data.data(), masks.data(), out.data(), data.size());
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        total_ms += elapsed.count();

        if (run == 0) {
            uint64_t run_checksum = 0;
            for (size_t i = 0; i < out.size(); ++i) {
                run_checksum ^= out[i];
            }
            checksum = run_checksum;
        }
    }

    return {total_ms / static_cast<double>(runs), checksum};
}

template <typename Fn>
static std::pair<double, uint64_t> run_select_bench_avg(const std::vector<uint64_t> &blocks,
                                                         const std::vector<int> &ks,
                                                         Fn fn,
                                                         size_t runs = 5) {
    double total_ms = 0.0;
    uint64_t checksum = 0;

    for (size_t run = 0; run < runs; ++run) {
        uint64_t run_checksum = 0;
        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < blocks.size(); ++i) {
            run_checksum ^= fn(blocks[i], ks[i], i);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        total_ms += elapsed.count();

        if (run == 0) {
            checksum = run_checksum;
        }
    }

    return {total_ms / static_cast<double>(runs), checksum};
}

int main() {
    constexpr size_t ITERATIONS = 10'000'000;
    std::vector<uint64_t> data(ITERATIONS);
    std::vector<uint64_t> masks(ITERATIONS);
    std::vector<uint64_t> blocks(ITERATIONS);
    std::vector<int> ks(ITERATIONS);
    std::vector<zp7_masks_64_t> precomputed_ppp(ITERATIONS);

    std::mt19937_64 gen(std::random_device{}());
    for (size_t i = 0; i < ITERATIONS; ++i) {
        data[i] = gen();
        masks[i] = gen();

        uint64_t block = gen();
        block |= 1ULL;  // guarantee at least one set bit
        blocks[i] = block;

        const int pop = __builtin_popcountll(block);
        ks[i] = static_cast<int>(gen() % static_cast<uint64_t>(pop));
        precomputed_ppp[i] = zp7_ppp_64(block);
    }

    const auto [pdep_zp7_ms, pdep_zp7_ck] = run_bench_avg(data, masks, [](uint64_t a, uint64_t m) {
        return zp7_pdep_64(a, m);
    });
    const auto [pext_zp7_ms, pext_zp7_ck] = run_bench_avg(data, masks, [](uint64_t a, uint64_t m) {
        return zp7_pext_64(a, m);
    });

    std::cout << "--- ZP7 PDEP/PEXT Benchmark (10 Million Iterations) ---\n";
    std::cout << "ZP7 PDEP Time: " << pdep_zp7_ms << " ms\n";
    std::cout << "ZP7 PEXT Time: " << pext_zp7_ms << " ms\n";

#if HAS_BMI2
    const auto [pdep_native_ms, pdep_native_ck] = run_bench_avg(data, masks, [](uint64_t a, uint64_t m) {
        return _pdep_u64(a, m);
    });
    const auto [pext_native_ms, pext_native_ck] = run_bench_avg(data, masks, [](uint64_t a, uint64_t m) {
        return _pext_u64(a, m);
    });

    std::cout << "Native PDEP (_pdep_u64) Time: " << pdep_native_ms << " ms\n";
    std::cout << "Native PEXT (_pext_u64) Time: " << pext_native_ms << " ms\n";
    std::cout << "PDEP speedup (ZP7/native): " << (pdep_zp7_ms / pdep_native_ms) << "x\n";
    std::cout << "PEXT speedup (ZP7/native): " << (pext_zp7_ms / pext_native_ms) << "x\n";
    std::cout << "Checksums PDEP (ZP7/native): " << pdep_zp7_ck << " / " << pdep_native_ck << "\n";
    std::cout << "Checksums PEXT (ZP7/native): " << pext_zp7_ck << " / " << pext_native_ck << "\n";
#else
    std::cout << "Native BMI2 comparisons are disabled on this architecture/build (e.g., ARM/no __BMI2__).\n";
    std::cout << "Checksums PDEP (ZP7): " << pdep_zp7_ck << "\n";
    std::cout << "Checksums PEXT (ZP7): " << pext_zp7_ck << "\n";
#endif

#if HAS_SIMD_PDEP_PEXT
    std::cout << "\n--- " << SIMD_ARCH_NAME << " Batched PDEP/PEXT Benchmark (Variables Masks) ---\n";
    const auto [pdep_simd_ms, pdep_simd_ck] = run_bench_avg_batch(data, masks, [](const uint64_t* d, const uint64_t* m, uint64_t* out, size_t n) {
        simd_utils::simd_pdep_u64_batch(d, m, out, n);
    });
    const auto [pext_simd_ms, pext_simd_ck] = run_bench_avg_batch(data, masks, [](const uint64_t* d, const uint64_t* m, uint64_t* out, size_t n) {
        simd_utils::simd_pext_u64_batch(d, m, out, n);
    });

    std::cout << "Batched " << SIMD_ARCH_NAME << " PDEP Time: " << pdep_simd_ms << " ms\n";
    std::cout << "Batched " << SIMD_ARCH_NAME << " PEXT Time: " << pext_simd_ms << " ms\n";
    std::cout << "PDEP speedup (Batched " << SIMD_ARCH_NAME << " / Scalar ZP7): " << (pdep_zp7_ms / pdep_simd_ms) << "x\n";
    std::cout << "PEXT speedup (Batched " << SIMD_ARCH_NAME << " / Scalar ZP7): " << (pext_zp7_ms / pext_simd_ms) << "x\n";
    std::cout << "Checksums PDEP (Batched " << SIMD_ARCH_NAME << "): " << pdep_simd_ck << " (matches ZP7: " << (pdep_zp7_ck == pdep_simd_ck ? "Yes" : "No") << ")\n";
    std::cout << "Checksums PEXT (Batched " << SIMD_ARCH_NAME << "): " << pext_simd_ck << " (matches ZP7: " << (pext_zp7_ck == pext_simd_ck ? "Yes" : "No") << ")\n";
    
    std::cout << "\n--- " << SIMD_ARCH_NAME << " Batched PEXT Benchmark (Shared Mask) ---\n";
    uint64_t shared_mask = 0x0F0F0F0F0F0F0F0F;
    const auto [pext_scalar_shared_ms, pext_scalar_shared_ck] = run_bench_avg(data, masks, [shared_mask](uint64_t a, uint64_t /*m*/) {
        return zp7_pext_64(a, shared_mask);
    });
    const auto [pext_simd_shared_ms, pext_simd_shared_ck] = run_bench_avg_batch(data, masks, [shared_mask](const uint64_t* d, const uint64_t* /*m*/, uint64_t* out, size_t n) {
        simd_utils::simd_pext_u64_shared_mask(d, shared_mask, out, n);
    });

    std::cout << "Scalar ZP7 PEXT (Shared Mask) Time: " << pext_scalar_shared_ms << " ms\n";
    std::cout << "Batched " << SIMD_ARCH_NAME << " PEXT (Shared Mask) Time: " << pext_simd_shared_ms << " ms\n";
    std::cout << "Shared Mask PEXT speedup (Batched " << SIMD_ARCH_NAME << " / Scalar ZP7): " << (pext_scalar_shared_ms / pext_simd_shared_ms) << "x\n";
    std::cout << "Checksums PEXT (Shared vs Batched): " << pext_scalar_shared_ck << " / " << pext_simd_shared_ck << " (Matches: " << (pext_scalar_shared_ck == pext_simd_shared_ck ? "Yes" : "No") << ")\n";
#else
    std::cout << "\n--- SIMD Batched Benchmarks Skipped (No supported architecture) ---\n";
#endif

    std::cout << "\n--- Select Benchmark (k-th set bit in 64-bit block) ---\n";

    const auto [sel_zp7_ms, sel_zp7_ck] = run_select_bench_avg(blocks, ks, [](uint64_t block, int k, size_t) {
        const uint64_t nth_bit_mask = 1ULL << k;
        const uint64_t deposited = zp7_pdep_64(nth_bit_mask, block);
        return static_cast<uint64_t>(__builtin_ctzll(deposited));
    });
    const auto [sel_zp7_cached_ms, sel_zp7_cached_ck] = run_select_bench_avg(blocks, ks, [&](uint64_t, int k, size_t i) {
        return zp7_select_pre_64(static_cast<uint64_t>(k), &precomputed_ppp[i]);
    });
    const auto [sel_swar_lut_ms, sel_swar_lut_ck] = run_select_bench_avg(blocks, ks, [](uint64_t block, int k, size_t) {
        return _select64(block, k);
    });
    const auto [sel_swar_no_lut_ms, sel_swar_no_lut_ck] = run_select_bench_avg(blocks, ks, [](uint64_t block, int k, size_t) {
        return _select_64(block, k);
    });

    std::cout << "ZP7-based select time:        " << sel_zp7_ms << " ms\n";
    std::cout << "ZP7 select (cached PPP):      " << sel_zp7_cached_ms << " ms\n";
    std::cout << "SWAR select (with LUT) time:  " << sel_swar_lut_ms << " ms\n";
    std::cout << "SWAR select (no LUT) time:    " << sel_swar_no_lut_ms << " ms\n";

#if HAS_BMI2
    const auto [sel_native_ms, sel_native_ck] = run_select_bench_avg(blocks, ks, [](uint64_t block, int k, size_t) {
        const uint64_t nth_bit_mask = 1ULL << k;
        const uint64_t deposited = _pdep_u64(nth_bit_mask, block);
        return static_cast<uint64_t>(__builtin_ctzll(deposited));
    });

    std::cout << "Native BMI2 select time:      " << sel_native_ms << " ms\n";
    std::cout << "Speedup (ZP7/SWAR LUT):       " << (sel_zp7_ms / sel_swar_lut_ms) << "x\n";
    std::cout << "Speedup (ZP7 cached/SWAR LUT):" << (sel_zp7_cached_ms / sel_swar_lut_ms) << "x\n";
    std::cout << "Speedup (ZP7/SWAR no LUT):    " << (sel_zp7_ms / sel_swar_no_lut_ms) << "x\n";
    std::cout << "Speedup (ZP7 cached/native):  " << (sel_zp7_cached_ms / sel_native_ms) << "x\n";
    std::cout << "Speedup (SWAR LUT/native):    " << (sel_swar_lut_ms / sel_native_ms) << "x\n";
    std::cout << "Speedup (SWAR no LUT/native): " << (sel_swar_no_lut_ms / sel_native_ms) << "x\n";
    std::cout << "Checksums select (ZP7/ZP7cached/native/LUT/noLUT): "
              << sel_zp7_ck << " / " << sel_zp7_cached_ck << " / " << sel_native_ck << " / "
              << sel_swar_lut_ck << " / " << sel_swar_no_lut_ck << "\n";
#else
    std::cout << "Native BMI2 select comparison skipped on this architecture/build.\n";
    std::cout << "Speedup (ZP7/SWAR LUT):       " << (sel_zp7_ms / sel_swar_lut_ms) << "x\n";
    std::cout << "Speedup (ZP7 cached/SWAR LUT):" << (sel_zp7_cached_ms / sel_swar_lut_ms) << "x\n";
    std::cout << "Speedup (ZP7/SWAR no LUT):    " << (sel_zp7_ms / sel_swar_no_lut_ms) << "x\n";
    std::cout << "Checksums select (ZP7/ZP7cached/LUT/noLUT): " << sel_zp7_ck << " / "
              << sel_zp7_cached_ck << " / " << sel_swar_lut_ck << " / " << sel_swar_no_lut_ck << "\n";
#endif

    return 0;
}