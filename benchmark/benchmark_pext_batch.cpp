#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <numeric>

#include "../hashtable/hashtable.h"
#include "../segment/segment.h"
#include "../fingerprint_gen_helper/fingerprint_gen_helper.h"
#include "../config/config.h"

int main() {
    generate_hashtable(ht1, signatures_h1, important_bits_h1, indices_h1, arr_h1);
    
    constexpr size_t FP_index = 12;
    Segment<TestDefaultTraits> sg(FP_index);
    const auto ssdLog = std::make_unique<SSDLog<TestDefaultTraits>>("segment_bench_pext.txt", 100);

    std::vector<TestDefaultTraits::KEY_TYPE> keys;
    std::vector<TestDefaultTraits::VALUE_TYPE> values;
    
    std::mt19937_64 setup_gen(42);
    for (size_t i = 0; i < 150; ++i) { // Enough elements to densely populate segment without expanding too much
        TestDefaultTraits::KEY_TYPE key = setup_gen();
        TestDefaultTraits::VALUE_TYPE value = i;
        keys.push_back(key);
        values.push_back(value);
        auto pt = ssdLog->write(key, value);
        auto hash_val = Hashing<TestDefaultTraits>::hash_digest(key);
        sg.write(hash_val, *ssdLog.get(), pt);
    }

    // Find any valid slot mapping that hits HT1 to center our batched queries on
    size_t target_block = -1;
    size_t target_lslot = -1;
    bool found_valid_slot = false;
    
    for (auto k : keys) {
        auto test_fp = Hashing<TestDefaultTraits>::hash_digest(k);
        uint32_t t_block = test_fp.range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS * 2, FP_index - COUNT_SLOT_BITS);
        uint32_t t_slot = test_fp.range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS, FP_index);
        
        auto ctx = sg.blockList[t_block].prepare_slot(t_slot, FP_index);
        if (ctx.use_ht && !ctx.ten_one && ctx.slot_occupied) {
            target_block = t_block;
            target_lslot = t_slot;
            found_valid_slot = true;
            std::cout << "Successfully found HT populated slot: " << t_slot << " in block: " << t_block << std::endl;
            break;
        }
    }
    
    if (!found_valid_slot) {
        std::cerr << "Setup failed: Could not find any HT populated slots in the loaded segment." << std::endl;
        return 1;
    }
    
    // Filter the keys to those only belonging to our chosen valid slot
    std::vector<TestDefaultTraits::KEY_TYPE> target_keys;
    for (auto k : keys) {
        auto test_fp = Hashing<TestDefaultTraits>::hash_digest(k);
        uint32_t t_block = test_fp.range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS * 2, FP_index - COUNT_SLOT_BITS);
        uint32_t t_slot = test_fp.range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS, FP_index);
        if (t_block == target_block && t_slot == target_lslot) {
            target_keys.push_back(k);
        }
    }
    
    // Generate workload
    constexpr size_t NUM_QUERIES = 2'000'000;
    std::vector<BitsetWrapper<FINGERPRINT_SIZE>> queries(NUM_QUERIES);
    std::mt19937_64 gen(1234);
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        int idx = gen() % target_keys.size();
        queries[i] = Hashing<TestDefaultTraits>::hash_digest(target_keys[idx]);
    }
    
    std::vector<std::optional<TestDefaultTraits::ENTRY_TYPE>> results(NUM_QUERIES);
    
    // Benchmark scalar original read
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<NUM_QUERIES; ++i) {
        results[i] = sg.read(queries[i], *ssdLog.get());
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto ms_scalar = std::chrono::duration_cast<std::chrono::milliseconds>(end_scalar - start_scalar).count();
    
    // Benchmark Two Phase Batched
    std::vector<std::optional<TestDefaultTraits::ENTRY_TYPE>> results_batch(NUM_QUERIES);
    auto start_batch = std::chrono::high_resolution_clock::now();
    sg.read_batch_pext(queries.data(), results_batch.data(), NUM_QUERIES, *ssdLog.get());
    auto end_batch = std::chrono::high_resolution_clock::now();
    auto ms_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    
    // Verification
    for(size_t i=0; i<NUM_QUERIES; ++i) {
        if (!results[i].has_value() || !results_batch[i].has_value() || 
            results[i].value().value != results_batch[i].value().value) {
            std::cerr << "Mismatch at " << i << ". Scalar read: " << (results[i].has_value() ? results[i].value().value : -1) 
                      << " Batch read: " << (results_batch[i].has_value() ? results_batch[i].value().value : -1) << std::endl;
            return 1;
        }
    }
    
    std::cout << "--- Two-Phase Slot Resolution vs Regular get_index ---" << std::endl;
    std::cout << "Architecture SIMD capability active: " << SIMD_ARCH_NAME << std::endl;
    std::cout << "Queries: " << NUM_QUERIES << std::endl;
    std::cout << "Regular read() time: " << ms_scalar << " ms" << std::endl;
    std::cout << "Two-Phase read_batch_pext() time (including grouping overhead): " << ms_batch << " ms" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(ms_scalar) / static_cast<double>(ms_batch) << "x" << std::endl;
    
    return 0;
}
