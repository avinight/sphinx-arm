// ZP7 (Zach's Peppy Parallel-Prefix-Popcountin' PEXT/PDEP Polyfill)
//
// Copyright (c) 2020 Zach Wegner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#   define ZP7_ALWAYS_INLINE __attribute__((always_inline)) inline
#   define ZP7_LIKELY(x) __builtin_expect(!!(x), 1)
#   define ZP7_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#   define ZP7_ALWAYS_INLINE inline
#   define ZP7_LIKELY(x) (x)
#   define ZP7_UNLIKELY(x) (x)
#endif

#if defined(__x86_64__) || defined(__i386__)
#   include <immintrin.h>
#elif defined(__aarch64__) || defined(__arm__)
#   include <arm_neon.h>
#endif

// Auto-enable optional fast paths when the compiler target supports them.
#if !defined(HAS_CLMUL) && (defined(__PCLMUL__) || defined(__PCLMULQDQ__))
#   define HAS_CLMUL
#endif

#if !defined(HAS_BZHI) && defined(__BMI2__)
#   define HAS_BZHI
#endif

#if !defined(HAS_POPCNT) && defined(__POPCNT__)
#   define HAS_POPCNT
#endif

// ZP7: branchless PEXT/PDEP replacement code for non-Intel processors
//
// The PEXT/PDEP instructions are pretty cool, with various (usually arcane)
// uses, behaving like bitwise gather/scatter instructions. They were introduced
// by Intel with the BMI2 instructions on Haswell.
//
// AMD processors implement these instructions, but very slowly. PEXT/PDEP can
// take from 18 to ~300 cycles, depending on the input mask. See this table:
// https://mobile.twitter.com/InstLatX64/status/1209095219087585281
// Other processors don't have PEXT/PDEP at all. This code is a polyfill for
// these processors. It's much slower than the raw instructions on Intel chips
// (which are 3L1T), but should be faster than AMD's implementation.
//
// Description of the algorithm
// ====
//
// This code uses a "parallel prefix popcount" technique (hereafter PPP for
// brevity). What this means is that we determine, for every bit in the input
// mask, how many bits below it are set. Or rather, aren't set--we need to get
// a count of how many bits each input bit should be shifted to get to its final
// position, that is, the difference between the bit-index of its destination
// and its original bit-index. This is the same as the count of unset bits in
// the mask below each input bit.
//
// The dumb way to get this PPP would be to create a 64-element array in a loop,
// but we want to do this in a bit-parallel fashion. So we store the counts
// "vertically" across six 64-bit values: one 64-bit value holds bit 0 of each
// of the 64 counts, another holds bit 1, etc. We can compute these counts
// fairly easily using a parallel prefix XOR (XOR is equivalent to a 1-bit
// adder that wraps around and ignores carrying). Using parallel prefix XOR as
// a 1-bit adder, we can build an n-bit adder by shifting the result left by
// one and ANDing with the input bits: this computes the carry by seeing where
// an input bit causes the 1-bit sum to overflow from 1 to 0. The shift left
// is needed anyways, because we want the PPP values to represent population
// counts *below* each bit, not including the bit itself.
//
// For processors with the CLMUL instructions (most x86 CPUs since ~2010), we
// can do the parallel prefix XOR and left shift in one instruction, by
// doing a carry-less multiply by -2. This is enabled with the HAS_CLMUL define.
//
// Anyways, once we have these six 64-bit values of the PPP, we can use each
// PPP bit to shift input bits by a power of two. That is, input bits that are
// in the bit-0 PPP mask are shifted by 2**0==1, bits in the bit-1 mask get
// shifted by 2, and so on, for shifts by 4, 8, 16, and 32 bits. Out of these
// six shifts, any shift value between 0 and 63 can be composed.
//
// For PEXT, we have to perform each shift in increasing order (1, 2, ...32) so
// that input bits don't overlap in the intermediate results. PDEP is the
// opposite: the 32-bit shift needs to happen first to make room for the smaller
// shifts. There's also a small complication for PDEP in that the PPP values
// line up where the input bits *end*, rather than where the input bits start
// like for PEXT. This means each bit mask needs to be shifted backwards before
// ANDing with the input bits.
//
// For both PEXT/PDEP the input bits need to be pre-masked so that only the
// relevant bits are being shifted around. For PEXT, this is a simple AND
// (input &= mask), but for PDEP we have to mask out everything but the low N
// bits, where N is the population count of the mask.

#define N_BITS      (6)

typedef struct {
    uint64_t mask;
    uint64_t ppp_bit[N_BITS];
    uint8_t byte_prefix[9];
} zp7_masks_64_t;

#if defined(__aarch64__) || defined(__arm__)
typedef struct {
    uint8_t popcnt[256];
    uint8_t compress[256][256];
    uint8_t deposit[256][256];
    uint8_t select[256][8];
} zp7_pext_lut_t;

static ZP7_ALWAYS_INLINE uint8_t zp7_pext8_fallback(uint8_t a, uint8_t mask) {
    uint8_t out = 0;
    uint8_t out_bit = 0;
    while (mask) {
        const uint8_t lsb = static_cast<uint8_t>(mask & static_cast<uint8_t>(-mask));
        if (a & lsb) {
            out |= static_cast<uint8_t>(1u << out_bit);
        }
        ++out_bit;
        mask = static_cast<uint8_t>(mask & static_cast<uint8_t>(mask - 1));
    }
    return out;
}

static ZP7_ALWAYS_INLINE uint8_t zp7_pdep8_fallback(uint8_t a, uint8_t mask) {
    uint8_t out = 0;
    uint8_t in_bit = 0;
    while (mask) {
        const uint8_t lsb = static_cast<uint8_t>(mask & static_cast<uint8_t>(-mask));
        if (a & static_cast<uint8_t>(1u << in_bit)) {
            out = static_cast<uint8_t>(out | lsb);
        }
        ++in_bit;
        mask = static_cast<uint8_t>(mask & static_cast<uint8_t>(mask - 1));
    }
    return out;
}

static inline const zp7_pext_lut_t &zp7_get_pext_lut() {
    static const zp7_pext_lut_t lut = [] {
        zp7_pext_lut_t t{};
        for (unsigned m = 0; m < 256; ++m) {
            uint8_t mm = static_cast<uint8_t>(m);
            uint8_t pc = 0;
            for (uint8_t x = mm; x; x = static_cast<uint8_t>(x & static_cast<uint8_t>(x - 1))) {
                ++pc;
            }
            t.popcnt[m] = pc;
            for (unsigned a = 0; a < 256; ++a) {
                t.compress[m][a] = zp7_pext8_fallback(static_cast<uint8_t>(a), mm);
                t.deposit[m][a] = zp7_pdep8_fallback(static_cast<uint8_t>(a), mm);
            }

            uint8_t bit_index = 0;
            for (unsigned bit = 0; bit < 8; ++bit) {
                if (mm & static_cast<uint8_t>(1u << bit)) {
                    t.select[m][bit_index++] = static_cast<uint8_t>(bit);
                }
            }
            for (; bit_index < 8; ++bit_index) {
                t.select[m][bit_index] = 8;
            }
        }
        return t;
    }();
    return lut;
}
#endif

#ifndef HAS_CLMUL
// If we don't have access to the CLMUL instruction, emulate it with
// shifts and XORs
static ZP7_ALWAYS_INLINE uint64_t prefix_sum(uint64_t x) {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    return x;
}
#endif

#ifndef HAS_POPCNT
#if defined(__aarch64__)
// POPCNT polyfill. ARMv8 has a vcnt instruction that counts bits in parallel across a 64-bit lane, so we can use that.
static ZP7_ALWAYS_INLINE uint64_t popcnt_64(uint64_t x) {
    uint8x8_t v = vcnt_u8(vcreate_u8(x));
    return vaddv_u8(v);
}

#else
// POPCNT polyfill. See this page for information about the algorithm:
// https://www.chessprogramming.org/Population_Count#SWAR-Popcount
static ZP7_ALWAYS_INLINE uint64_t popcnt_64(uint64_t x) {
    // Match the SWAR form used in bitset_wrapper.h
    constexpr uint64_t L8 = 0x0101010101010101ULL;
    uint64_t s = x - ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    s = (s & 0x3333333333333333ULL) + ((s >> 2) & 0x3333333333333333ULL);
    s = ((s + (s >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * L8;
    return s >> 56;
}
#endif
#endif

// Parallel-prefix-popcount. This is shared by both PEXT and PDEP.
// The cached byte_prefix data is also used by the select helper below.
static ZP7_ALWAYS_INLINE zp7_masks_64_t zp7_ppp_64(uint64_t mask) {
    zp7_masks_64_t r{};
    r.mask = mask;

    // Count *unset* bits
    mask = ~mask;

    // Cache byte-level prefix popcounts for fast select queries.
    // byte_prefix[i] = number of set bits in bytes [0, i).
    uint8_t prefix = 0;
    r.byte_prefix[0] = 0;
    for (unsigned i = 0; i < 8; ++i) {
    #if defined(__GNUC__) || defined(__clang__)
        prefix = static_cast<uint8_t>(prefix + __builtin_popcountll((r.mask >> (i * 8)) & 0xFFULL));
    #elif defined(_MSC_VER)
        prefix = static_cast<uint8_t>(prefix + __popcnt64((r.mask >> (i * 8)) & 0xFFULL));
    #else
        prefix = static_cast<uint8_t>(prefix + popcnt_64((r.mask >> (i * 8)) & 0xFFULL));
    #endif
        r.byte_prefix[i + 1] = prefix;
    }

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
    // ARMv8 Crypto Extension: Polynomial Multiply Long
    uint64_t m = mask;
    uint64_t neg_2 = -2ULL; // 0xFFFFFFFFFFFFFFFE

    for (int i = 0; i < N_BITS - 1; i++) {
        // Polynomial multiply over GF(2) is identical to carry-less multiply
        poly128_t p_128 = vmull_p64((poly64_t)m, (poly64_t)neg_2);
        
        // Extract the lower 64 bits from the 128-bit result vector
        uint64_t bit = vgetq_lane_u64(vreinterpretq_u64_p128(p_128), 0);
        
        r.ppp_bit[i] = bit;
        m &= bit;
    }
    r.ppp_bit[N_BITS - 1] = -m << 1;

#elif defined(HAS_CLMUL)
    // Move the mask and -2 to XMM registers for CLMUL
    __m128i m = _mm_cvtsi64_si128(mask);
    __m128i neg_2 = _mm_cvtsi64_si128(-2LL);
    for (int i = 0; i < N_BITS - 1; i++) {
        // Do a 1-bit parallel prefix popcount, shifted left by 1,
        // in one carry-less multiply by -2.
        __m128i bit = _mm_clmulepi64_si128(m, neg_2, 0);
        r.ppp_bit[i] = _mm_cvtsi128_si64(bit);

        // Get the carry bit of the 1-bit parallel prefix popcount. On
        // the next iteration, we will sum this bit to get the next mask
        m = _mm_and_si128(m, bit);
    }
    // For the last iteration, we can use a regular multiply by -2 instead of a
    // carry-less one (or rather, a strength reduction of that, with
    // neg/add/etc), since there can't be any carries anyways. That is because
    // the last value of m (which has one bit set for every 32nd unset mask bit)
    // has at most two bits set in it, when mask is zero and thus there are 64
    // bits set in ~mask. If two bits are set, one of them is the top bit, which
    // gets shifted out, since we're counting bits below each mask bit.
    r.ppp_bit[N_BITS - 1] = -_mm_cvtsi128_si64(m) << 1;
#else
    for (int i = 0; i < N_BITS - 1; i++) {
        // Do a 1-bit parallel prefix popcount, shifted left by 1
        uint64_t bit = prefix_sum(mask << 1);
        r.ppp_bit[i] = bit;

        // Get the carry bit of the 1-bit parallel prefix popcount. On
        // the next iteration, we will sum this bit to get the next mask
        mask &= bit;
    }
    // The last iteration won't carry, so just use neg/shift. See the CLMUL
    // case above for justification.
    r.ppp_bit[N_BITS - 1] = -mask << 1;
#endif

    return r;
}

// ---------------------------------------------------------------------------
// PEXT helpers
// ---------------------------------------------------------------------------

static ZP7_ALWAYS_INLINE uint64_t zp7_pext_pre_64(uint64_t a, const zp7_masks_64_t *masks) {
    // Mask only the bits that are set in the input mask. Otherwise they collide
    // with input bits and screw everything up
    a &= masks->mask;

    for (int i = 0; i < N_BITS; ++i) {
        const uint64_t shift = 1ULL << i;
        const uint64_t bit = masks->ppp_bit[i];
        a = (a & ~bit) | ((a & bit) >> shift);
    }

    return a;
}

#if defined(__aarch64__) || defined(__arm__)
static ZP7_ALWAYS_INLINE uint64_t zp7_pext_arm_64(uint64_t a, uint64_t mask) {
    // ARM fast path: 8-bit PEXT LUT + byte-wise packing.
    const zp7_pext_lut_t &lut = zp7_get_pext_lut();

    uint64_t out = 0;
    unsigned shift = 0;
    for (unsigned i = 0; i < 8; ++i) {
        const uint8_t m8 = static_cast<uint8_t>(mask >> (i * 8));
        const uint8_t a8 = static_cast<uint8_t>(a >> (i * 8));
        out |= static_cast<uint64_t>(lut.compress[m8][a8]) << shift;
        shift += lut.popcnt[m8];
    }

    return out;
}
#endif

static ZP7_ALWAYS_INLINE uint64_t zp7_select_pre_64(uint64_t rank, const zp7_masks_64_t *masks) {
#if defined(__aarch64__) || defined(__arm__)
    const zp7_pext_lut_t &lut = zp7_get_pext_lut();
    const uint64_t mask = masks->mask;
    unsigned byte = 0;

    byte |= static_cast<unsigned>(rank >= masks->byte_prefix[4]) << 2;
    byte |= static_cast<unsigned>(rank >= masks->byte_prefix[byte + 2]) << 1;
    byte |= static_cast<unsigned>(rank >= masks->byte_prefix[byte + 1]);

    const uint8_t byte_mask = static_cast<uint8_t>(mask >> (byte * 8));
    const uint8_t rank_in_byte = static_cast<uint8_t>(rank - masks->byte_prefix[byte]);
    return static_cast<uint64_t>((byte * 8) + lut.select[byte_mask][rank_in_byte]);
#else
    const uint64_t nth_bit_mask = 1ULL << rank;
    const uint64_t deposited = zp7_pdep_pre_64(nth_bit_mask, masks);
    return static_cast<uint64_t>(__builtin_ctzll(deposited));
#endif
}

static ZP7_ALWAYS_INLINE uint64_t zp7_pext_64(uint64_t a, uint64_t mask) {
    // Common degenerate masks can avoid PPP setup entirely.
    if (ZP7_UNLIKELY(mask == 0ULL)) {
        return 0ULL;
    }
    if (ZP7_UNLIKELY(mask == ~0ULL)) {
        return a;
    }
    // If mask is contiguous low bits (2^k - 1), PEXT is just an AND.
    if (ZP7_UNLIKELY((mask & (mask + 1ULL)) == 0ULL)) {
        return a & mask;
    }

#if defined(__aarch64__) || defined(__arm__)
    return zp7_pext_arm_64(a, mask);
#endif

    // Fused one-shot PEXT: compute PPP bits and apply shifts immediately.
    // This avoids writing/reading a temporary zp7_masks_64_t on hot paths.
    a &= mask;
    uint64_t m = ~mask;

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO)
    const uint64_t neg_2 = -2ULL;

    for (int i = 0; i < N_BITS - 1; ++i) {
        const uint64_t shift = 1ULL << i;
        const poly128_t p = vmull_p64((poly64_t)m, (poly64_t)neg_2);
        const uint64_t bit = vgetq_lane_u64(vreinterpretq_u64_p128(p), 0);
        a = (a & ~bit) | ((a & bit) >> shift);
        m &= bit;
    }

    const uint64_t b_last = -m << 1;
    a = (a & ~b_last) | ((a & b_last) >> (1ULL << (N_BITS - 1)));
    return a;

#elif defined(HAS_CLMUL)
    __m128i m128 = _mm_cvtsi64_si128(m);
    const __m128i neg_2 = _mm_cvtsi64_si128(-2LL);

    for (int i = 0; i < N_BITS - 1; ++i) {
        const uint64_t shift = 1ULL << i;
        const __m128i p = _mm_clmulepi64_si128(m128, neg_2, 0);
        const uint64_t bit = _mm_cvtsi128_si64(p);
        a = (a & ~bit) | ((a & bit) >> shift);
        m128 = _mm_and_si128(m128, p);
    }

    const uint64_t b_last = -_mm_cvtsi128_si64(m128) << 1;
    a = (a & ~b_last) | ((a & b_last) >> (1ULL << (N_BITS - 1)));
    return a;

#else
    for (int i = 0; i < N_BITS - 1; ++i) {
        const uint64_t shift = 1ULL << i;
        const uint64_t bit = prefix_sum(m << 1);
        a = (a & ~bit) | ((a & bit) >> shift);
        m &= bit;
    }

    const uint64_t b_last = -m << 1;
    a = (a & ~b_last) | ((a & b_last) >> (1ULL << (N_BITS - 1)));
    return a;
#endif
}

// ---------------------------------------------------------------------------
// PDEP helpers
// ---------------------------------------------------------------------------

static ZP7_ALWAYS_INLINE uint64_t zp7_pdep_pre_64(uint64_t a, const zp7_masks_64_t *masks) {
#if defined(__GNUC__) || defined(__clang__)
    uint64_t popcnt = static_cast<uint64_t>(__builtin_popcountll(masks->mask));
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    uint64_t popcnt = static_cast<uint64_t>(__popcnt64(masks->mask));
#else
    uint64_t popcnt = popcnt_64(masks->mask);
#endif

    // Mask just the bits that will end up in the final result--the low P bits,
    // where P is the popcount of the mask. The other bits would collide.
    // We need special handling for the mask==-1 case: because 64-bit shifts are
    // implicitly modulo 64 on x86 (and (uint64_t)1 << 64 is technically
    // undefined behavior in C), the regular "a &= (1 << pop) - 1" doesn't
    // work: (1 << popcnt(-1)) - 1 == (1 << 64) - 1 == (1 << 0) - 1 == 0, but
    // this should be -1. The BZHI instruction (introduced with BMI2, the same
    // instructions as PEXT/PDEP) handles this properly, but isn't portable.

#ifdef HAS_BZHI
    a = _bzhi_u64(a, popcnt);
#elif defined(__aarch64__)
    // THEORETICAL BEST FOR ARM (M1/AArch64)
    // 1. Constant Width: If the compiler knows 'popcnt' at compile-time (via inlining),
    //    it completely deletes this logic and emits a single 1-cycle `UBFX` or `AND` instruction.
    // 2. Variable Width: If 'popcnt' is dynamic, the ternary operator forces the compiler
    //    to emit a branchless `CMP` + `CSEL` (Conditional Select) instruction pair.
    //    This safely bypasses the 64-bit shift wrap-around bug in just ~3 cycles.
    a &= (popcnt == 64) ? ~0ULL : ((1ULL << popcnt) - 1);
#else
    // If we don't have BZHI, use a portable workaround.  Since (mask == -1)
    // is equivalent to popcnt(mask) >> 6, use that to mask out the 1 << 64
    // case.
    uint64_t pop_mask = (1ULL << popcnt) & ~(popcnt >> 6);
    a &= pop_mask - 1;
#endif

    // For each bit in the PPP, shift left only those bits that are set in
    // that bit's mask. We do this in the opposite order compared to PEXT
    for (int i = N_BITS - 1; i >= 0; i--) {
        uint64_t shift = 1 << i;
        uint64_t bit = masks->ppp_bit[i] >> shift;
        // Micro-optimization: the bits that get shifted and those that don't
        // will always be disjoint, so we can add them instead of ORing them.
        // The shifts of 1 and 2 can thus merge with the adds to become LEAs.
        a = (a & ~bit) + ((a & bit) << shift);
    }
    return a;
}

#if defined(__aarch64__) || defined(__arm__)
static ZP7_ALWAYS_INLINE uint64_t zp7_pdep_arm_64(uint64_t a, uint64_t mask) {
    if (ZP7_UNLIKELY(mask == 0ULL)) {
        return 0ULL;
    }
    if (ZP7_UNLIKELY(mask == ~0ULL)) {
        return a;
    }

    const zp7_pext_lut_t &lut = zp7_get_pext_lut();

    uint64_t in = a;
    uint64_t out = 0;

    for (unsigned i = 0; i < 8; ++i) {
        const uint8_t m8 = static_cast<uint8_t>(mask >> (i * 8));
        out |= static_cast<uint64_t>(lut.deposit[m8][static_cast<uint8_t>(in)]) << (i * 8);
        in >>= lut.popcnt[m8];
    }

    return out;
}
#endif

static ZP7_ALWAYS_INLINE uint64_t zp7_pdep_64(uint64_t a, uint64_t mask) {
#if defined(__aarch64__) || defined(__arm__)
    return zp7_pdep_arm_64(a, mask);
#endif

    zp7_masks_64_t masks = zp7_ppp_64(mask);
    return zp7_pdep_pre_64(a, &masks);
}