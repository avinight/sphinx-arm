#pragma once

#include <stdint.h>
#include <stddef.h>

#if defined(__AVX2__)
#define HAS_SIMD_PDEP_PEXT 1
#define SIMD_ARCH_NAME "AVX2"
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define HAS_SIMD_PDEP_PEXT 1
#define SIMD_ARCH_NAME "NEON"
#include <arm_neon.h>
#else
#define HAS_SIMD_PDEP_PEXT 0
#define SIMD_ARCH_NAME "NONE"
#endif

namespace simd_utils {

// Batched AVX2 PEXT for 64-bit values (processes 4 uint64_t at a time)
// Adapted from http://0x80.pl/notesen/2025-01-05-simd-pdep-pext.html
#if defined(__AVX2__)
template <int MAX_MASK_BITS = 64, bool EARLY_EXIT = true>
inline void simd_pext_u64_batch(const uint64_t* data_arr, const uint64_t* mask_arr, uint64_t* out_arr, size_t n) {
    static_assert(MAX_MASK_BITS > 0);
    static_assert(MAX_MASK_BITS <= 32);
    
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i zero = _mm256_setzero_si256();
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data_arr[i]));
        __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_arr[i]));
        __m256i out = zero;
        __m256i bit = one;
        
        for (int j = 0; j < MAX_MASK_BITS; j++) {
            const __m256i m0 = _mm256_sub_epi64(mask, one);
            const __m256i m1 = _mm256_and_si256(mask, m0);
            const __m256i m2 = _mm256_xor_si256(mask, m1);
            
            const __m256i d0 = _mm256_and_si256(data, m2);
            const __m256i d_eq_0 = _mm256_cmpeq_epi64(d0, zero);
            const __m256i o0 = _mm256_andnot_si256(d_eq_0, bit);
            
            out = _mm256_or_si256(out, o0);
            mask = m1;
            bit = _mm256_add_epi64(bit, bit); 
            
            if (EARLY_EXIT && _mm256_testc_si256(zero, mask)) break;
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_arr[i]), out);
    }
    
    for (; i < n; ++i) {
#if defined(__BMI2__)
        out_arr[i] = _pext_u64(data_arr[i], mask_arr[i]);
#else
        out_arr[i] = 0;
        uint64_t d = data_arr[i];
        uint64_t m = mask_arr[i];
        uint64_t b = 1;
        while (m) {
            uint64_t m_lsb = m & -m;
            if (d & m_lsb) out_arr[i] |= b;
            m &= m - 1;
            b <<= 1;
        }
#endif
    }
}

template <int MAX_MASK_BITS = 64, bool EARLY_EXIT = true>
inline void simd_pext_u64_shared_mask(const uint64_t* data_arr, uint64_t shared_mask_val, uint64_t* out_arr, size_t n) {
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i shared_mask = _mm256_set1_epi64x(shared_mask_val);
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data_arr[i]));
        __m256i mask = shared_mask;
        __m256i out = zero;
        __m256i bit = one;
        
        for (int j = 0; j < MAX_MASK_BITS; j++) {
            const __m256i m0 = _mm256_sub_epi64(mask, one);
            const __m256i m1 = _mm256_and_si256(mask, m0);
            const __m256i m2 = _mm256_xor_si256(mask, m1);
            
            const __m256i d0 = _mm256_and_si256(data, m2);
            const __m256i d_eq_0 = _mm256_cmpeq_epi64(d0, zero);
            const __m256i o0 = _mm256_andnot_si256(d_eq_0, bit);
            
            out = _mm256_or_si256(out, o0);
            mask = m1;
            bit = _mm256_add_epi64(bit, bit);
            
            if (EARLY_EXIT && _mm256_testc_si256(zero, mask)) break;
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_arr[i]), out);
    }
    
    for (; i < n; ++i) {
#if defined(__BMI2__)
        out_arr[i] = _pext_u64(data_arr[i], shared_mask_val);
#else
        out_arr[i] = 0;
        uint64_t d = data_arr[i];
        uint64_t m = shared_mask_val;
        uint64_t b = 1;
        while (m) {
            uint64_t m_lsb = m & -m;
            if (d & m_lsb) out_arr[i] |= b;
            m &= m - 1;
            b <<= 1;
        }
#endif
    }
}

template <int MAX_DATA_BITS = 64, bool EARLY_EXIT = true>
inline void simd_pdep_u64_batch(const uint64_t* data_arr, const uint64_t* mask_arr, uint64_t* out_arr, size_t n) {
    const __m256i one = _mm256_set1_epi64x(1);
    const __m256i zero = _mm256_setzero_si256();
    
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data_arr[i]));
        __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&mask_arr[i]));
        __m256i out = zero;
        __m256i bit = one;
        
        for (int j = 0; j < MAX_DATA_BITS; j++) {
            const __m256i m0 = _mm256_sub_epi64(mask, one);
            const __m256i m1 = _mm256_and_si256(mask, m0);
            const __m256i m2 = _mm256_xor_si256(mask, m1);
            
            const __m256i d0 = _mm256_and_si256(data, bit);
            const __m256i d1 = _mm256_cmpeq_epi64(d0, zero);
            const __m256i m3 = _mm256_andnot_si256(d1, m2);
            
            out = _mm256_or_si256(out, m3);
            mask = m1;
            bit = _mm256_add_epi64(bit, bit);
            
            if (EARLY_EXIT && _mm256_testc_si256(zero, mask)) break;
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_arr[i]), out);
    }
    
    for (; i < n; ++i) {
#if defined(__BMI2__)
        out_arr[i] = _pdep_u64(data_arr[i], mask_arr[i]);
#else
        out_arr[i] = 0;
        uint64_t d = data_arr[i];
        uint64_t m = mask_arr[i];
        uint64_t b = 1;
        while (m) {
            uint64_t m_lsb = m & -m;
            if (d & b) out_arr[i] |= m_lsb;
            m &= m - 1;
            b <<= 1;
        }
#endif
    }
}

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)

template <int MAX_MASK_BITS = 64, bool EARLY_EXIT = true>
inline void simd_pext_u64_batch(const uint64_t* data_arr, const uint64_t* mask_arr, uint64_t* out_arr, size_t n) {
    const uint64x2_t one = vdupq_n_u64(1);
    const uint64x2_t zero = vdupq_n_u64(0);
    
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t data = vld1q_u64(&data_arr[i]);
        uint64x2_t mask = vld1q_u64(&mask_arr[i]);
        uint64x2_t out = zero;
        uint64x2_t bit = one;
        
        for (int j = 0; j < MAX_MASK_BITS; j++) {
            const uint64x2_t m0 = vsubq_u64(mask, one);
            const uint64x2_t m1 = vandq_u64(mask, m0);
            const uint64x2_t m2 = veorq_u64(mask, m1);
            
            const uint64x2_t d0 = vandq_u64(data, m2);
            const uint64x2_t d_eq_0 = vceqq_u64(d0, zero);
            // vbicq(a, b) => a & ~b
            const uint64x2_t o0 = vbicq_u64(bit, d_eq_0);
            
            out = vorrq_u64(out, o0);
            mask = m1;
            bit = vaddq_u64(bit, bit);
            
            if (EARLY_EXIT && vgetq_lane_u64(mask, 0) == 0 && vgetq_lane_u64(mask, 1) == 0) break;
        }
        vst1q_u64(&out_arr[i], out);
    }
    
    for (; i < n; ++i) {
        out_arr[i] = 0;
        uint64_t d = data_arr[i];
        uint64_t m = mask_arr[i];
        uint64_t b = 1;
        while (m) {
            uint64_t m_lsb = m & -m;
            if (d & m_lsb) out_arr[i] |= b;
            m &= m - 1;
            b <<= 1;
        }
    }
}

template <int MAX_MASK_BITS = 64, bool EARLY_EXIT = true>
inline void simd_pext_u64_shared_mask(const uint64_t* data_arr, uint64_t shared_mask_val, uint64_t* out_arr, size_t n) {
    const uint64x2_t one = vdupq_n_u64(1);
    const uint64x2_t zero = vdupq_n_u64(0);
    const uint64x2_t shared_mask = vdupq_n_u64(shared_mask_val);
    
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t data = vld1q_u64(&data_arr[i]);
        uint64x2_t mask = shared_mask;
        uint64x2_t out = zero;
        uint64x2_t bit = one;
        
        for (int j = 0; j < MAX_MASK_BITS; j++) {
            const uint64x2_t m0 = vsubq_u64(mask, one);
            const uint64x2_t m1 = vandq_u64(mask, m0);
            const uint64x2_t m2 = veorq_u64(mask, m1);
            
            const uint64x2_t d0 = vandq_u64(data, m2);
            const uint64x2_t d_eq_0 = vceqq_u64(d0, zero);
            const uint64x2_t o0 = vbicq_u64(bit, d_eq_0);
            
            out = vorrq_u64(out, o0);
            mask = m1;
            bit = vaddq_u64(bit, bit);
            
            if (EARLY_EXIT && vgetq_lane_u64(mask, 0) == 0 && vgetq_lane_u64(mask, 1) == 0) break;
        }
        vst1q_u64(&out_arr[i], out);
    }
    
    for (; i < n; ++i) {
        out_arr[i] = 0;
        uint64_t d = data_arr[i];
        uint64_t m = shared_mask_val;
        uint64_t b = 1;
        while (m) {
            uint64_t m_lsb = m & -m;
            if (d & m_lsb) out_arr[i] |= b;
            m &= m - 1;
            b <<= 1;
        }
    }
}

template <int MAX_DATA_BITS = 64, bool EARLY_EXIT = true>
inline void simd_pdep_u64_batch(const uint64_t* data_arr, const uint64_t* mask_arr, uint64_t* out_arr, size_t n) {
    const uint64x2_t one = vdupq_n_u64(1);
    const uint64x2_t zero = vdupq_n_u64(0);
    
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        uint64x2_t data = vld1q_u64(&data_arr[i]);
        uint64x2_t mask = vld1q_u64(&mask_arr[i]);
        uint64x2_t out = zero;
        uint64x2_t bit = one;
        
        for (int j = 0; j < MAX_DATA_BITS; j++) {
            const uint64x2_t m0 = vsubq_u64(mask, one);
            const uint64x2_t m1 = vandq_u64(mask, m0);
            const uint64x2_t m2 = veorq_u64(mask, m1);
            
            const uint64x2_t d0 = vandq_u64(data, bit);
            const uint64x2_t d1 = vceqq_u64(d0, zero);
            const uint64x2_t m3 = vbicq_u64(m2, d1);
            
            out = vorrq_u64(out, m3);
            mask = m1;
            bit = vaddq_u64(bit, bit);
            
            if (EARLY_EXIT && vgetq_lane_u64(mask, 0) == 0 && vgetq_lane_u64(mask, 1) == 0) break;
        }
        vst1q_u64(&out_arr[i], out);
    }
    
    for (; i < n; ++i) {
        out_arr[i] = 0;
        uint64_t d = data_arr[i];
        uint64_t m = mask_arr[i];
        uint64_t b = 1;
        while (m) {
            uint64_t m_lsb = m & -m;
            if (d & b) out_arr[i] |= m_lsb;
            m &= m - 1;
            b <<= 1;
        }
    }
}

#endif

} // namespace simd_utils
