#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>  // For std::memcmp
#include <cstdlib>  // For std::abs
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../zp7/zp7.h"
// true = use zp7_pdep_64, false = use _select64
static constexpr bool kUseZp7Polyfill = false;

#if (defined(__x86_64__) || defined(__i386__)) && defined(__BMI2__) && (defined(__GNUC__) || defined(__clang__))
#   include <x86intrin.h> // For _pdep_u64 and _tzcnt_u64
#else
#   include <arm_neon.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && defined(__BMI2__) && (defined(__GNUC__) || defined(__clang__))
    static constexpr bool kHasBmi2 = true;
#else
    static constexpr bool kHasBmi2 = false;
#endif

const uint8_t kSelectInByte[2048] = {
	8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
	1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
	2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
	1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0,
	1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
	2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
	1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0,
	1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 8, 8, 1,
	8, 2, 2, 1, 8, 3, 3, 1, 3, 2, 2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2,
	2, 1, 8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1,
	4, 3, 3, 1, 3, 2, 2, 1, 8, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4,
	4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1,
	3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 7, 7, 1, 7, 2,
	2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
	7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3,
	3, 1, 3, 2, 2, 1, 7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1,
	4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2,
	2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 8, 8, 8, 8, 8, 8, 2,
	8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3, 4, 3, 3, 2, 8, 8,
	8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3,
	4, 3, 3, 2, 8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4,
	4, 2, 6, 4, 4, 3, 4, 3, 3, 2, 8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2,
	6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 7, 8, 7, 7, 2, 8, 7,
	7, 3, 7, 3, 3, 2, 8, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2, 8, 7, 7, 5,
	7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3,
	3, 2, 8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2,
	6, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5,
	5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3, 8, 8, 8, 8, 8, 8,
	8, 5, 8, 8, 8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3,
	8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6,
	6, 4, 6, 4, 4, 3, 8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5,
	6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7,
	7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3, 8, 8, 8, 7, 8, 7, 7, 5,
	8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3, 8, 8,
	8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4,
	6, 4, 4, 3, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5,
	5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 4, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6,
	6, 4, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6, 8, 6, 6, 5,
	8, 6, 6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8,
	8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7,
	8, 7, 7, 5, 8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4, 8, 8, 8, 8, 8, 8,
	8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4,
	8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6,
	6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6,
	8, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
	8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 8,
	8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6,
	6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7
};

/**
 * Returns the position of the `k`-th 1 in the 64-bit word x. `k` is 0-based,
 * so `k`=0 returns the position of the first 1.
 *
 * Uses the broadword selection algorithm by Vigna [1], improved by Gog and
 * Petri [2] and Vigna [3].
 *
 * [1] Sebastiano Vigna. Broadword Implementation of Rank/Select Queries. WEA,
 * 2008
 *
 * [2] Simon Gog, Matthias Petri. Optimized succinct data structures for
 * massive data. Softw. Pract. Exper., 2014
 *
 * [3] Sebastiano Vigna. MG4J 5.2.1. http://mg4j.di.unimi.it/
 * The following code is taken from
 * https://github.com/facebook/folly/blob/b28186247104f8b90cfbe094d289c91f9e413317/folly/experimental/Select64.h
 *
 * @param x - The mask to select from.
 * @param k - The desired bit rank.
 * @returns The position of the `k`-th set bit.
 */
static inline uint64_t _select64(uint64_t x, int k)
{
	if (k >= __builtin_popcountll(x)) { return 64; }

	const uint64_t kOnesStep4  = 0x1111111111111111ULL;
	const uint64_t kOnesStep8  = 0x0101010101010101ULL;
	const uint64_t kMSBsStep8  = 0x80ULL * kOnesStep8;

	uint64_t s = x;
	s = s - ((s & 0xA * kOnesStep4) >> 1);
	s = (s & 0x3 * kOnesStep4) + ((s >> 2) & 0x3 * kOnesStep4);
	s = (s + (s >> 4)) & 0xF * kOnesStep8;
	uint64_t byteSums = s * kOnesStep8;

	uint64_t kStep8 = k * kOnesStep8;
	uint64_t geqKStep8 = (((kStep8 | kMSBsStep8) - byteSums) & kMSBsStep8);
	uint64_t place = __builtin_popcountll(geqKStep8) * 8;
	uint64_t byteRank = k - (((byteSums << 8) >> place) & (uint64_t)(0xFF));
	return place + kSelectInByte[((x >> place) & 0xFF) | (byteRank << 8)];
}

constexpr uint64_t L8 = 0x0101010101010101ULL; // Lowest bit of each byte
constexpr uint64_t H8 = 0x8080808080808080ULL; // Highest bit of each byte

/**
 * @brief Parallel unsigned <= (byte-wise).
 * High bit in each byte is set if x_i <= y_i (unsigned).
 */
inline constexpr uint64_t u_le8(uint64_t x, uint64_t y) {
    return ((((y | H8) - (x & ~H8)) | (x ^ y)) ^ (x & ~y)) & H8;
}

/**
 * @brief Parallel signed <= (byte-wise).
 * High bit in each byte is set if x_i <= y_i (signed).
 */
inline constexpr uint64_t i_le8(uint64_t x, uint64_t y) {
    return (((y | H8) - (x & ~H8)) ^ x ^ y) & H8;
}

/**
 * @brief Parallel > 0 (byte-wise).
 * High bit in each byte is set if x_i > 0.
 */
inline constexpr uint64_t u_nz8(uint64_t x) {
    return (((x | H8) - L8) | x) & H8;
}

static inline uint64_t _select_64(uint64_t x, int k) {
    // Broadword SWAR Fallback (Vigna's Algorithm)
    uint64_t s = word - ((word & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    s = (s & 0x3333333333333333ULL) + ((s >> 2) & 0x3333333333333333ULL);
    s = ((s + (s >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * L8;

    const uint64_t r = static_cast<uint64_t>(zero_based_rank);
    
    // Locate target byte
    uint64_t b = ((i_le8(s, r * L8) >> 7) * L8 >> 53) & ~7ULL;
    
    // Rank inside target byte
    const uint64_t l = r - (((s << 8) >> b) & 0xFFULL);
    
    // Isolate byte and compute cumulative sums
    s = (u_nz8((((word >> b) & 0xFFULL) * L8) & 0x8040201008040201ULL) >> 7) * L8;
    
    // Final bit index
    return static_cast<size_t>(b + (((i_le8(s, l * L8)) >> 7) * L8 >> 56));
}

constexpr size_t count_bits(size_t n, size_t count = static_cast<size_t>(-1)) {  // same as log2
    return n ? count_bits(n >> 1, count + 1) : count;
}

constexpr size_t REGISTER_SIZE = 64;
constexpr size_t REGISTER_SIZE_BITS = static_cast<size_t>(count_bits(REGISTER_SIZE));
// 128 here is a optimization it should be N

#define GET_INDEX(agg_index) ((agg_index) >> (REGISTER_SIZE_BITS))
#define GET_OFFSET(agg_index) ((agg_index) & (REGISTER_SIZE - 1))

inline uint64_t GET_ZERO_MSB(const size_t first_zero_index) {
    if (first_zero_index == 64)
        return ~0ULL;
    return (1ULL << first_zero_index) - 1;
}
inline uint64_t GET_ONE_MSB(const size_t first_one_index) {
    return ~GET_ZERO_MSB(first_one_index);
}

template <size_t N>
class BitsetWrapper {
   public:
    static constexpr size_t NUM_REGS = (N + REGISTER_SIZE - 1) / REGISTER_SIZE;

    uint64_t bitset[(N + REGISTER_SIZE - 1) / REGISTER_SIZE] = {};

    BitsetWrapper() = default;

    explicit BitsetWrapper(const std::string &binaryString) {
        setInputString(binaryString);
    }

    explicit BitsetWrapper(const std::vector<uint64_t>& vals, bool is_array) {
        if (vals.size() != NUM_REGS || !is_array) throw std::invalid_argument("not correct nums of args");
        for (size_t i = 0; i < NUM_REGS; ++i)
            this->bitset[i] = vals[i];
    }
    inline BitsetWrapper<N>& operator=(const BitsetWrapper<N>& other) {
        if (this != &other)
            std::memcpy(this->bitset, other.bitset, sizeof(bitset));
        return *this;
    }

    inline bool operator>(const BitsetWrapper& other) const {
        for (int i = 0; i < N; i++)
            if (get(i) != other.get(i))
                return get(i) > other.get(i);
        return false;
    }

    inline bool operator<(const BitsetWrapper& other) const {
        for (int i = 0; i < NUM_REGS; i++)
            if (bitset[i] != other.bitset[i])
                return bitset[i] < other.bitset[i];
        return false;
    }
    inline bool operator==(const BitsetWrapper &other) const {
        return std::memcmp(bitset, other.bitset, sizeof(bitset)) == 0;
    }
    inline bool operator!=(const BitsetWrapper &other) const {
        return !(*this == other);
    }
    inline BitsetWrapper operator^(const BitsetWrapper &other) const {
        BitsetWrapper result;
        for (size_t i = 0; i < NUM_REGS; i++)
            result.bitset[i] = this->bitset[i] ^ other.bitset[i];
        return result;
    }
    inline BitsetWrapper operator&(const BitsetWrapper &other) const {
        BitsetWrapper result;
        for (size_t i = 0; i < NUM_REGS; i++)
            result.bitset[i] = this->bitset[i] & other.bitset[i];
        return result;
    }
    inline BitsetWrapper operator~() const {
        BitsetWrapper result;
        for (size_t i = 0; i < NUM_REGS; i++)
            result.bitset[i] = ~this->bitset[i];
        return result;
    }
    void setInputString(const std::string &binaryString) {
        for (size_t i = 0; i < binaryString.size() && i < N; ++i) {
            if (binaryString[i] == '1') {
                set(i, true);
            } else if (binaryString[i] == '0') {
                set(i, false);
            } else {
                throw std::invalid_argument("Input string contains invalid characters (not 0 or 1).");
            }
        }
    }

    void setInputInt64(const int64_t repeating_number) {
        for (size_t j = 0; j < NUM_REGS; j++) {
            set_fast_one_reg(j, 0, REGISTER_SIZE, static_cast<size_t>(repeating_number));
        }
    }

    [[nodiscard]] std::string getInputString(size_t firstInvalidIndex = size()) const {
        std::string s;
        for (size_t i = 0; i < firstInvalidIndex; i++) {
            s += (get(i) ? '1' : '0');
        }
        return s;
    }

    inline static size_t __attribute__((always_inline)) size() {
        return N;
    }

    [[nodiscard]] inline size_t __attribute__((always_inline)) get_leading_zeros(size_t reg_idx) const {
        return __builtin_clzll(bitset[reg_idx]);
    }

    // need to be tested
    [[nodiscard]] inline size_t __attribute__((always_inline)) get_second_leading_zeros(size_t reg_idx) const {
        int64_t nVal = static_cast<int64_t>(~(1ULL << (REGISTER_SIZE - __builtin_clzll(bitset[reg_idx]) - 1)) & bitset[reg_idx]);
        if (nVal == 0)
            return REGISTER_SIZE;
        return __builtin_clzll(static_cast<unsigned long long>(nVal));
    }

    inline size_t __attribute__((always_inline)) get_trailing_zeros(size_t reg_idx) {
        return __builtin_ctzll(bitset[reg_idx]);
    }

    inline bool __attribute__((always_inline)) get(const size_t index, const size_t from = 0) const {
        const size_t agg_index = index + from;
        if (__builtin_expect(agg_index >= N, 0))
            throw std::invalid_argument("index out of bound 1");
        const size_t idx = GET_INDEX(agg_index), offset = GET_OFFSET(agg_index);
        return static_cast<bool>(bitset[idx] & (1ULL << offset));
    }
    inline int get_first_one_before_slow(const size_t index, const size_t from = 0) const {
        const size_t agg_index = index + from;
        if (__builtin_expect(agg_index >= N, 0))
            throw std::out_of_range("index out of bound 2");

        // Start from the given index and move backwards
        for (int i = static_cast<int>(agg_index); i > 0; --i) {
            if (get(i - 1)) {
                return i - 1;  // Found the first '1' before the given index
            }
        }
        // no 1s before agg_index
        return -1;
    }

    inline size_t range(const size_t index1, const size_t index2) const {
        if (__builtin_expect((index2 < index1 || index2 > N), 0))
            throw std::invalid_argument("Invalid index range");
        size_t result = 0;
        // could be faster
        for (size_t i = index1; i < index2; i++) {
            result |= (get(i) << (index2 - i - 1));
        }
        return result;
    }

    inline size_t range_fast(const size_t index1, const size_t index2) const {
        if (__builtin_expect((index2 < index1 || index2 > N), 0))
            throw std::invalid_argument("Invalid index range");
        const size_t idx1 = GET_INDEX(index1);
        const size_t offset1 = GET_OFFSET(index1);
        const size_t idx2 = GET_INDEX(index2);
        const size_t offset2 = GET_OFFSET(index2);
        if (idx1 == idx2) {
            return ((bitset[idx1] >> offset1) & ((1ULL << (offset2 - offset1)) - 1));
        } else if (idx1 + 1 == idx2) {
            return ((bitset[idx1] >> offset1) & ((1ULL << (REGISTER_SIZE - offset1)) - 1)) + ((bitset[idx2] & ((1ULL << offset2) - 1)) << (REGISTER_SIZE - offset1));
        }
        throw std::invalid_argument("should not be here in range fast");
    }

    inline auto __attribute__((always_inline)) range_fast_one_reg(const size_t index, const size_t offset1, const size_t offset2) const {
        auto x = bitset[index] >> offset1;
        auto y = (1ULL << (offset2 - offset1)) - 1;
        auto z = (x & y);
        return z;
    }

    inline auto __attribute__((always_inline)) range_fast_2(const size_t index1, const size_t index2) const {
        const size_t idx1 = GET_INDEX(index1);
        const size_t offset1 = GET_OFFSET(index1);
        const size_t idx2 = GET_INDEX(index2);
        const size_t offset2 = GET_OFFSET(index2);
        if (idx1 == idx2) {
            auto x = bitset[idx1] >> offset1;
            auto y = (1ULL << (offset2 - offset1)) - 1;
            auto z = (x & y);
            return z;
        } else if (idx1 + 1 == idx2) {
            return ((bitset[idx1] >> offset1) & ((1ULL << (REGISTER_SIZE - offset1)) - 1)) + ((bitset[idx2] & ((1ULL << offset2) - 1)) << (REGISTER_SIZE - offset1));
        }
        throw std::invalid_argument("should not be here in range 2");
    }

    inline bool __attribute__((always_inline)) set(const size_t index, bool value) {
        if (__builtin_expect((index >= size()), 0))
            throw std::invalid_argument("index out of bound set 2");
        const size_t idx = GET_INDEX(index), offset = GET_OFFSET(index);
        if (idx * REGISTER_SIZE + offset >= size()) {
            return false;
        }
        if (value)
            bitset[idx] |= 1ULL << offset;
        else
            bitset[idx] &= ~(1ULL << offset);
        return true;
    }
    inline void __attribute__((always_inline)) set_fast_one_reg(const size_t idx, const size_t offset_start, const size_t offset_after_end, const size_t value) {
        // idx2 exclusive
        const size_t range_length = offset_after_end - offset_start;
        size_t range_mask = ((1UL << range_length) - 1) << offset_start;

        range_mask |= -(range_length == REGISTER_SIZE);

        bitset[idx] &= ~range_mask;
        bitset[idx] |= (value << offset_start) & range_mask;
    }

    inline void __attribute__((always_inline)) set_fast_two_reg(const size_t idx1, const size_t idx2, const size_t value) {
        // idx2 exclusive
        const size_t reg_i1 = GET_INDEX(idx1), offset1 = GET_OFFSET(idx1);
        const size_t reg_i2 = GET_INDEX(idx2), offset2 = GET_OFFSET(idx2);

        if (reg_i1 == reg_i2) {
            set_fast_one_reg(reg_i1, offset1, offset2, value);
        } else {
            set_fast_one_reg(reg_i1, offset1, REGISTER_SIZE, value);                   // Set bits in the first register
            set_fast_one_reg(reg_i2, 0, offset2, value >> (REGISTER_SIZE - offset1));  // Set bits in the second register
        }
    }
    void printBitset() const {
        for (size_t i = 0; i < size(); ++i) {
            std::cout << get(i);
            if (i % REGISTER_SIZE == REGISTER_SIZE - 1)
                std::cout << ' ';
        }
        std::cout << std::endl;
    }

    void printRange(size_t i, size_t j) {
        for (; i < j; ++i) {
            std::cout << get(i);
        }
        std::cout << std::endl;
    }

    size_t rank_dumb(const size_t pos) {
        size_t count = 0;
        for (size_t i = 0; i < pos; ++i) {
            if (get(i)) ++count;
        }
        return count;
    }

    size_t select_dumb(const size_t nth) const {
        if (nth == 0) throw std::out_of_range("nth must be greater than 0.");
        size_t count = 0;
        for (size_t i = 0; i < N; ++i) {
            if (get(i)) ++count;
            if (count == nth) return i;
        }
        return size();
    }

    inline size_t __attribute__((always_inline)) rank(const size_t pos) const {
        // exclusive the end
        const size_t index = GET_INDEX(pos), offset = GET_OFFSET(pos);
        size_t count = 0;
        for (size_t i = 0; i < index; ++i) {
            count += __builtin_popcountll(bitset[i]);
        }
        const uint64_t mask = (offset == 0) ? 0ULL : ((static_cast<uint64_t>(1) << offset) - 1);
        count += __builtin_popcountll(bitset[index] & mask);
        return count;
    }

    inline size_t __attribute__((always_inline)) select(const size_t nth, const size_t start_from_reg = 0) const {
        size_t count = 0;
        size_t blockPopCount = 0;
        size_t i;
        uint64_t block;
        for (i = start_from_reg; i < NUM_REGS && count + blockPopCount < nth; ++i) {
            count += blockPopCount;
            block = bitset[i];
            blockPopCount = __builtin_popcountll(block);
        }

        // 0-based index of the target bit within the isolated 64-bit block
        const size_t bitIndexInWord = nth - count - 1; 
        size_t bitPosition;

    #if kHasBmi2 // --- Fast Path: x86 Hardware BMI2 ---
        const uint64_t nthBitMask = static_cast<uint64_t>(1) << bitIndexInWord;
        const uint64_t depositMask = _pdep_u64(nthBitMask, block);
        bitPosition = _tzcnt_u64(depositMask);
    #else // --- Software Fallbacks (ARM / Non-BMI2 x86) ---
        if constexpr (kUseZp7Polyfill) {
            const uint64_t nthBitMask = static_cast<uint64_t>(1) << bitIndexInWord;
            const uint64_t depositMask = zp7_pdep_64(nthBitMask, block);
            bitPosition = __builtin_ctzll(depositMask); 
        } else {
            bitPosition = _select64(block, bitIndexInWord);
        }
    #endif
        return ((i - start_from_reg - 1) * REGISTER_SIZE) + bitPosition;
    }

    // at most in two adjacent registers
    inline std::pair<size_t, size_t> select_two(const size_t fromth, const size_t toth, const size_t start_from_reg = 0) const {
        size_t count = 0;
        size_t i1 = start_from_reg;
        uint64_t block = bitset[i1];
        uint64_t block_pop_count = __builtin_popcountll(block);

        while (i1 < NUM_REGS) {
            if (count + block_pop_count < fromth) {
                count += block_pop_count;
                block = bitset[++i1];
                block_pop_count = __builtin_popcountll(block);
                continue;
            }
            break;
        }

        // Upgraded to 1ULL to prevent 32-bit signed overflow on large shifts
        const uint64_t nth_fromth = (1ULL + (1ULL << (toth - fromth))) << (fromth - count - 1);
        
        size_t bit_pos_fromth;
        size_t bit_pos_toth_lz;

    #if kHasBmi2 // --- Fast Path: x86 Hardware BMI2 ---
        const uint64_t deposit_fromth = _pdep_u64(nth_fromth, block);
        bit_pos_fromth = _tzcnt_u64(deposit_fromth);
        bit_pos_toth_lz = _lzcnt_u64(deposit_fromth);
    #else // --- Software Fallbacks ---
        if constexpr (kUseZp7Polyfill) {
            const uint64_t deposit_fromth = zp7_pdep_64(nth_fromth, block);
            bit_pos_fromth = __builtin_ctzll(deposit_fromth);
            bit_pos_toth_lz = __builtin_clzll(deposit_fromth);
        } else {
            bit_pos_fromth = _select64(block, fromth - count - 1);
            
            // To maintain the `diff == 0` spill-over check, we need to simulate _lzcnt.
            // If the `toth` bit is out of bounds for this block, _select64 returns 64.
            size_t actual_toth_pos = _select64(block, toth - count - 1);
            if (actual_toth_pos == 64) {
                bit_pos_toth_lz = REGISTER_SIZE - 1 - bit_pos_fromth; // forces diff to 0
            } else {
                bit_pos_toth_lz = REGISTER_SIZE - 1 - actual_toth_pos;
            }
        }
    #endif
        size_t diff = REGISTER_SIZE - 1 - bit_pos_toth_lz - bit_pos_fromth;
        size_t fromth_return = ((i1 - start_from_reg) << REGISTER_SIZE_BITS) + bit_pos_fromth;
        size_t toth_return = fromth_return + diff;

        if (diff == 0) {  // very rare case aprox = 15/200
            count += block_pop_count;
            block = bitset[++i1];
            block_pop_count = __builtin_popcountll(block);
            
            const uint64_t nth_toth = 1ULL << (toth - count - 1);
            size_t bit_pos_toth;

        #if kHasBmi2
            const uint64_t deposit_toth = _pdep_u64(nth_toth, block);
            bit_pos_toth = _tzcnt_u64(deposit_toth);
        #else
            if constexpr (kUseZp7Polyfill) {
                const uint64_t deposit_toth = zp7_pdep_64(nth_toth, block);
                bit_pos_toth = __builtin_ctzll(deposit_toth);
            } else {
                bit_pos_toth = _select64(block, toth - count - 1);
            }
        #endif
            toth_return = ((i1 - start_from_reg) << REGISTER_SIZE_BITS) + bit_pos_toth;
        }

        return std::make_pair(fromth_return, toth_return);
    }

    inline std::pair<size_t, size_t> select2(const size_t nth, const size_t start_from_reg = 0) const {
        if (__builtin_expect(nth == 0, 0))
            throw std::out_of_range("nth must be greater than 0.");

        size_t count = 0;
        for (size_t i = start_from_reg; i < (N + REGISTER_SIZE - 1) >> REGISTER_SIZE_BITS; ++i) {
            const uint64_t block = bitset[i];
            const size_t blockPopCount = __builtin_popcountll(block);

            if (count + blockPopCount < nth) {
                count += blockPopCount;
                continue;
            }

            const uint64_t nthBitMask = static_cast<uint64_t>(3) << (nth - count - 1);  // Create a mask for the nth bit
            
            size_t bitPosition;
            size_t bitPosition_left;

        #if kHasBmi2
            const uint64_t depositMask = _pdep_u64(nthBitMask, block);
            bitPosition = _tzcnt_u64(depositMask);
            bitPosition_left = _lzcnt_u64(depositMask);
        #else
            if constexpr (kUseZp7Polyfill) {
                const uint64_t depositMask = zp7_pdep_64(nthBitMask, block);
                bitPosition = __builtin_ctzll(depositMask);
                bitPosition_left = __builtin_clzll(depositMask); 
            } else {
                // Fallback using _select64 twice if no PDEP is available at all
                bitPosition = _select64(block, nth - count - 1);
                bitPosition_left = _select64(block, nth - count);
            }
        #endif

            size_t diff = REGISTER_SIZE - 1 - bitPosition_left - bitPosition;
            return std::make_pair(((i - start_from_reg) << REGISTER_SIZE_BITS) + bitPosition,
                                  ((i - start_from_reg) << REGISTER_SIZE_BITS) + bitPosition + diff);
        }
        return std::make_pair(size(), 0);
    }
    inline void count_contiguous(size_t &index, int &count) const {
        // stop *after* the first 0
        while (!get(index++)) {
            count++;
        }
    }

    inline void count_contiguous_until_false(size_t &index, int &count) const {
        // stop *after* the first 0
        while (get(index++)) {
            count++;
        }
    }

    inline void deprecated_shift(int64_t steps, int64_t from, int64_t to = static_cast<int64_t>(size())) {
        // to exclusive / from inclusive
    #ifdef DEBUG
        if (!(to >= from && to <= static_cast<int64_t>(size()))) {
            throw std::invalid_argument("Invalid to/from range");
        }
    #endif
        if (steps == 0) return;

        if (steps > 0) {
            // Right shift
            for (auto i = to - 1; i >= from + steps; --i) {
                set(static_cast<size_t>(i), get(static_cast<size_t>(i - steps)));
            }
            // Clear the bits that have been shifted out
            for (auto i = from; i < from + std::min(steps, to - from); ++i) {
                set(static_cast<size_t>(i), false);
            }
        } else {
            // Left shift
            const int64_t abs_steps = -steps;
            for (auto i = from; i < to - abs_steps; ++i) {
                set(static_cast<size_t>(i), get(static_cast<size_t>(i + abs_steps)));
            }
            for (auto i = to - std::min(abs_steps, to - from); i < to; ++i) {
                set(static_cast<size_t>(i), false);
            }
        }
    }
// some ideas
// remove steps sign so remove one branch
    inline void shift_smart(int steps, size_t from, size_t to = size()) {
        const size_t bitShift = static_cast<size_t>(std::abs(steps));
    #ifdef DEBUG
        if (!(to >= from && to <= size())) {
            throw std::invalid_argument("Invalid to/from range");
        }
        if (bitShift >= REGISTER_SIZE) {
            throw std::invalid_argument("Shift exceeds register size");
        }
    #endif
    
        if (steps == 0 || to == from) return;
    
        const uint64_t startIdx = GET_INDEX(from);
        const uint64_t endIdx = GET_INDEX(to - 1);
        const uint64_t startOffset = GET_OFFSET(from);
        const uint64_t endOffset = GET_OFFSET(to - 1);
        const uint64_t oneOnMSBEnd = GET_ONE_MSB(endOffset + 1);
        const uint64_t zeroOnMSBStart = GET_ZERO_MSB(startOffset);
        const uint64_t to2end = bitset[endIdx] & oneOnMSBEnd;
        const uint64_t start2from = bitset[startIdx] & zeroOnMSBStart;

        if (steps > 0) {
            // Mask and compute spill over for startIdx
            const uint64_t startMask = ~zeroOnMSBStart;
            uint64_t maskedStart = bitset[startIdx] & startMask;

            // Shift elements from end to start
            for (int64_t i = static_cast<int64_t>(endIdx); i > static_cast<int64_t>(startIdx); --i) {
                const uint64_t prev = (i == static_cast<int64_t>(startIdx) + 1) ? maskedStart : bitset[i - 1];
                bitset[i] = (bitset[i] << bitShift) | (prev >> (REGISTER_SIZE - bitShift));
            }

            // Apply shifts to startIdx and restore preserved bits
            bitset[startIdx] = (maskedStart << bitShift) | start2from;
            bitset[endIdx] = (bitset[endIdx] & ~oneOnMSBEnd) | to2end;
        } else {
            // Mask and compute spill over for endIdx
            const uint64_t endMask = ~oneOnMSBEnd;
            uint64_t maskedEnd = bitset[endIdx] & endMask;
    
            // Shift elements from start to end
            for (int64_t i = static_cast<int64_t>(startIdx); i < static_cast<int64_t>(endIdx); ++i) {
                const uint64_t next = (i == static_cast<int64_t>(endIdx) - 1) ? maskedEnd : bitset[i + 1];
                bitset[i] = (bitset[i] >> bitShift) | (next << (REGISTER_SIZE - bitShift));
            }
    
            // Apply shifts to endIdx and restore preserved bits
            bitset[endIdx] = (maskedEnd >> bitShift) | to2end;
            bitset[startIdx] = (bitset[startIdx] & ~zeroOnMSBStart) | start2from;
        }
    }

    BitsetWrapper<N> replicateTrieStore() const {
        BitsetWrapper<N> copy;
        std::memcpy(copy.bitset, this->bitset, sizeof(this->bitset));
        return copy;
    }
};
