#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <arm_neon.h>

#define N_BITS (6)

struct zp7_masks_64_t {
    uint64_t mask;
    uint64_t ppp_bit[N_BITS];
};

// =====================================================================
// 1. SCALAR FALLBACK IMPLEMENTATION (No SIMD)
// =====================================================================
inline uint64_t prefix_sum(uint64_t x) {
    x ^= x << 1;  x ^= x << 2;  x ^= x << 4;
    x ^= x << 8;  x ^= x << 16; x ^= x << 32;
    return x;
}

zp7_masks_64_t zp7_ppp_64_scalar(uint64_t mask) {
    zp7_masks_64_t r; r.mask = mask; mask = ~mask;
    for (int i = 0; i < N_BITS - 1; i++) {
        uint64_t bit = prefix_sum(mask << 1);
        r.ppp_bit[i] = bit;
        mask &= bit;
    }
    r.ppp_bit[N_BITS - 1] = -mask << 1;
    return r;
}

uint64_t zp7_pdep_64_scalar(uint64_t a, uint64_t mask) {
    zp7_masks_64_t masks = zp7_ppp_64_scalar(mask);
    
    // Scalar POPCNT
    const uint64_t m_1 = 0x5555555555555555LLU;
    const uint64_t m_2 = 0x3333333333333333LLU;
    const uint64_t m_4 = 0x0f0f0f0f0f0f0f0fLLU;
    uint64_t x = masks.mask;
    x = x - ((x >> 1) & m_1);
    x = (x & m_2) + ((x >> 2) & m_2);
    x = (x + (x >> 4)) & m_4;
    uint64_t popcnt = (x * 0x0101010101010101LLU) >> 56;

    // Portable Workaround BZHI
    uint64_t pop_mask = (1ULL << popcnt) & ~(popcnt >> 6);
    a &= pop_mask - 1;

    for (int i = N_BITS - 1; i >= 0; i--) {
        uint64_t shift = 1 << i;
        uint64_t bit = masks.ppp_bit[i] >> shift;
        a = (a & ~bit) + ((a & bit) << shift);
    }
    return a;
}

// =====================================================================
// 2. ARM NEON OPTIMIZED IMPLEMENTATION
// =====================================================================
zp7_masks_64_t zp7_ppp_64_neon(uint64_t mask) {
    zp7_masks_64_t r; r.mask = mask; mask = ~mask;
    uint64_t m = mask;
    uint64_t neg_2 = -2ULL;

    for (int i = 0; i < N_BITS - 1; i++) {
        poly128_t p_128 = vmull_p64((poly64_t)m, (poly64_t)neg_2);
        uint64_t bit = vgetq_lane_u64(vreinterpretq_u64_p128(p_128), 0);
        r.ppp_bit[i] = bit;
        m &= bit;
    }
    r.ppp_bit[N_BITS - 1] = -m << 1;
    return r;
}

uint64_t zp7_pdep_64_neon(uint64_t a, uint64_t mask) {
    zp7_masks_64_t masks = zp7_ppp_64_neon(mask);
    
    // NEON POPCNT
    uint8x8_t v = vcnt_u8(vcreate_u8(masks.mask));
    uint64_t popcnt = vaddv_u8(v);

    // ARM CSEL Branchless BZHI
    a &= (popcnt == 64) ? ~0ULL : ((1ULL << popcnt) - 1);

    for (int i = N_BITS - 1; i >= 0; i--) {
        uint64_t shift = 1 << i;
        uint64_t bit = masks.ppp_bit[i] >> shift;
        a = (a & ~bit) + ((a & bit) << shift);
    }
    return a;
}

// =====================================================================
// 3. BENCHMARK HARNESS
// =====================================================================
int main() {
    const size_t ITERATIONS = 10000000;
    std::vector<uint64_t> data(ITERATIONS);
    std::vector<uint64_t> masks(ITERATIONS);

    // Generate random data to prevent branch prediction from cheating
    std::mt19937_64 gen(std::random_device{}());
    for (size_t i = 0; i < ITERATIONS; i++) {
        data[i] = gen();
        masks[i] = gen();
    }

    // Benchmark SCALAR
    uint64_t scalar_accumulator = 0;
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        scalar_accumulator ^= zp7_pdep_64_scalar(data[i], masks[i]);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> scalar_ms = end_scalar - start_scalar;

    // Benchmark NEON
    uint64_t neon_accumulator = 0;
    auto start_neon = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITERATIONS; i++) {
        neon_accumulator ^= zp7_pdep_64_neon(data[i], masks[i]);
    }
    auto end_neon = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> neon_ms = end_neon - start_neon;

    // Output Results
    std::cout << "--- PDEP Benchmark (10 Million Iterations) ---\n";
    std::cout << "Scalar Fallback Time: " << scalar_ms.count() << " ms\n";
    std::cout << "NEON Optimized Time:  " << neon_ms.count() << " ms\n";
    std::cout << "Speedup:              " << scalar_ms.count() / neon_ms.count() << "x\n";
    
    // Print accumulators to ensure the compiler doesn't optimize the loops away
    std::cout << "(Checksums: " << scalar_accumulator << ", " << neon_accumulator << ")\n";

    return 0;
}