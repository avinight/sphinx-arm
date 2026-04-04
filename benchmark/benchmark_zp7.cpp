#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "../bitset_wrapper/bitset_wrapper.h"
#include "../zp7/zp7.h"

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
static std::pair<double, uint64_t> run_bench(const std::vector<uint64_t> &data,
                                             const std::vector<uint64_t> &masks,
                                             Fn fn) {
    uint64_t checksum = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        checksum ^= fn(data[i], masks[i]);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return {elapsed.count(), checksum};
}

template <typename Fn>
static std::pair<double, uint64_t> run_select_bench(const std::vector<uint64_t> &blocks,
                                                    const std::vector<int> &ks,
                                                    Fn fn) {
    uint64_t checksum = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < blocks.size(); ++i) {
        checksum ^= fn(blocks[i], ks[i], i);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return {elapsed.count(), checksum};
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

    const auto [pdep_zp7_ms, pdep_zp7_ck] = run_bench(data, masks, [](uint64_t a, uint64_t m) {
        return zp7_pdep_64(a, m);
    });
    const auto [pext_zp7_ms, pext_zp7_ck] = run_bench(data, masks, [](uint64_t a, uint64_t m) {
        return zp7_pext_64(a, m);
    });

    std::cout << "--- ZP7 PDEP/PEXT Benchmark (10 Million Iterations) ---\n";
    std::cout << "ZP7 PDEP Time: " << pdep_zp7_ms << " ms\n";
    std::cout << "ZP7 PEXT Time: " << pext_zp7_ms << " ms\n";

#if HAS_BMI2
    const auto [pdep_native_ms, pdep_native_ck] = run_bench(data, masks, [](uint64_t a, uint64_t m) {
        return _pdep_u64(a, m);
    });
    const auto [pext_native_ms, pext_native_ck] = run_bench(data, masks, [](uint64_t a, uint64_t m) {
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

    std::cout << "\n--- Select Benchmark (k-th set bit in 64-bit block) ---\n";

    const auto [sel_zp7_ms, sel_zp7_ck] = run_select_bench(blocks, ks, [](uint64_t block, int k, size_t) {
        const uint64_t nth_bit_mask = 1ULL << k;
        const uint64_t deposited = zp7_pdep_64(nth_bit_mask, block);
        return static_cast<uint64_t>(__builtin_ctzll(deposited));
    });
    const auto [sel_zp7_cached_ms, sel_zp7_cached_ck] = run_select_bench(blocks, ks, [&](uint64_t, int k, size_t i) {
        const uint64_t nth_bit_mask = 1ULL << k;
        const uint64_t deposited = zp7_pdep_pre_64(nth_bit_mask, &precomputed_ppp[i]);
        return static_cast<uint64_t>(__builtin_ctzll(deposited));
    });
    const auto [sel_swar_lut_ms, sel_swar_lut_ck] = run_select_bench(blocks, ks, [](uint64_t block, int k, size_t) {
        return _select64(block, k);
    });
    const auto [sel_swar_no_lut_ms, sel_swar_no_lut_ck] = run_select_bench(blocks, ks, [](uint64_t block, int k, size_t) {
        return _select_64(block, k);
    });

    std::cout << "ZP7-based select time:        " << sel_zp7_ms << " ms\n";
    std::cout << "ZP7 select (cached PPP):      " << sel_zp7_cached_ms << " ms\n";
    std::cout << "SWAR select (with LUT) time:  " << sel_swar_lut_ms << " ms\n";
    std::cout << "SWAR select (no LUT) time:    " << sel_swar_no_lut_ms << " ms\n";

#if HAS_BMI2
    const auto [sel_native_ms, sel_native_ck] = run_select_bench(blocks, ks, [](uint64_t block, int k, size_t) {
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