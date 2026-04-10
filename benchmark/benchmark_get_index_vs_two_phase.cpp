
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <optional>

#include "../hashtable/hashtable.h"
#include "../segment/segment.h"
#include "../fingerprint_gen_helper/fingerprint_gen_helper.h"
#include "../config/config.h"

int main() {
	generate_hashtable(ht1, signatures_h1, important_bits_h1, indices_h1, arr_h1);

	constexpr size_t FP_index = 12;
	Segment<TestDefaultTraits> sg(FP_index);
	const auto ssdLog = std::make_unique<SSDLog<TestDefaultTraits>>("segment_bench_get_index_two_phase.txt", 100);

	std::vector<TestDefaultTraits::KEY_TYPE> keys;
	std::vector<TestDefaultTraits::VALUE_TYPE> values;

	std::mt19937_64 setup_gen(42);
	for (size_t i = 0; i < 150; ++i) {
		TestDefaultTraits::KEY_TYPE key = setup_gen();
		TestDefaultTraits::VALUE_TYPE value = i;
		keys.push_back(key);
		values.push_back(value);
		auto pt = ssdLog->write(key, value);
		auto hash_val = Hashing<TestDefaultTraits>::hash_digest(key);
		sg.write(hash_val, *ssdLog.get(), pt);
	}

	size_t target_block = static_cast<size_t>(-1);
	size_t target_lslot = static_cast<size_t>(-1);
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
			std::cout << "Found HT populated slot: " << t_slot << " in block: " << t_block << std::endl;
			break;
		}
	}

	if (!found_valid_slot) {
		std::cerr << "Setup failed: Could not find any HT populated slots in the loaded segment." << std::endl;
		return 1;
	}

	std::vector<TestDefaultTraits::KEY_TYPE> target_keys;
	for (auto k : keys) {
		auto test_fp = Hashing<TestDefaultTraits>::hash_digest(k);
		uint32_t t_block = test_fp.range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS * 2, FP_index - COUNT_SLOT_BITS);
		uint32_t t_slot = test_fp.range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS, FP_index);
		if (t_block == target_block && t_slot == target_lslot) {
			target_keys.push_back(k);
		}
	}

	if (target_keys.empty()) {
		std::cerr << "Setup failed: No keys mapped to the chosen slot." << std::endl;
		return 1;
	}

	constexpr size_t NUM_QUERIES = 2'000'000;
	std::vector<BitsetWrapper<FINGERPRINT_SIZE>> queries(NUM_QUERIES);
	std::mt19937_64 gen(1234);
	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		size_t idx = gen() % target_keys.size();
		queries[i] = Hashing<TestDefaultTraits>::hash_digest(target_keys[idx]);
	}

	auto &blk = sg.blockList[target_block];
	const auto ctx = blk.prepare_slot(target_lslot, FP_index);

	std::vector<std::optional<TestDefaultTraits::ENTRY_TYPE>> results_batch(NUM_QUERIES);
	std::vector<std::optional<TestDefaultTraits::ENTRY_TYPE>> results_batch_pext(NUM_QUERIES);
	std::vector<std::optional<TestDefaultTraits::ENTRY_TYPE>> results_prepared(NUM_QUERIES);

	std::vector<uint8_t> offsets_get_index(NUM_QUERIES);
	std::vector<uint8_t> offsets_zp7(NUM_QUERIES);
	std::vector<uint8_t> offsets_simd(NUM_QUERIES);
	std::vector<uint8_t> offsets_prepare_simd(NUM_QUERIES);
	std::uint64_t sum_get_index = 0;
	std::uint64_t sum_zp7 = 0;
	std::uint64_t sum_simd = 0;
	std::uint64_t sum_prepare_simd = 0;

	auto start_get_index = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		auto idx_pair = blk.get_index(queries[i], FP_index);
		offsets_get_index[i] = static_cast<uint8_t>(idx_pair.second);
		sum_get_index += offsets_get_index[i];
	}
	auto end_get_index = std::chrono::high_resolution_clock::now();
	auto ms_get_index = std::chrono::duration_cast<std::chrono::milliseconds>(end_get_index - start_get_index).count();

	auto start_zp7 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		offsets_zp7[i] = static_cast<uint8_t>(blk.resolve_offset_zp7(ctx, queries[i], FP_index));
		sum_zp7 += offsets_zp7[i];
	}
	auto end_zp7 = std::chrono::high_resolution_clock::now();
	auto ms_zp7 = std::chrono::duration_cast<std::chrono::milliseconds>(end_zp7 - start_zp7).count();

	auto start_simd = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		offsets_simd[i] = static_cast<uint8_t>(blk.resolve_offset(ctx, queries[i], FP_index));
		sum_simd += offsets_simd[i];
	}
	auto end_simd = std::chrono::high_resolution_clock::now();
	auto ms_simd = std::chrono::duration_cast<std::chrono::milliseconds>(end_simd - start_simd).count();

	auto start_prepare_simd = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		auto slot_index = queries[i].range_fast_one_reg(0, FP_index - COUNT_SLOT_BITS, FP_index);
		auto ctx_local = blk.prepare_slot(slot_index, FP_index);
		offsets_prepare_simd[i] = static_cast<uint8_t>(blk.resolve_offset(ctx_local, queries[i], FP_index));
		sum_prepare_simd += offsets_prepare_simd[i];
	}
	auto end_prepare_simd = std::chrono::high_resolution_clock::now();
	auto ms_prepare_simd = std::chrono::duration_cast<std::chrono::milliseconds>(end_prepare_simd - start_prepare_simd).count();

	auto start_prepared = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		results_prepared[i] = sg.read_prepared(queries[i], *ssdLog.get());
	}
	auto end_prepared = std::chrono::high_resolution_clock::now();
	auto ms_prepared = std::chrono::duration_cast<std::chrono::milliseconds>(end_prepared - start_prepared).count();

	auto start_batch = std::chrono::high_resolution_clock::now();
	sg.read_batch(queries.data(), results_batch.data(), NUM_QUERIES, *ssdLog.get());
	auto end_batch = std::chrono::high_resolution_clock::now();
	auto ms_batch = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();

	auto start_batch_pext = std::chrono::high_resolution_clock::now();
	sg.read_batch_pext(queries.data(), results_batch_pext.data(), NUM_QUERIES, *ssdLog.get());
	auto end_batch_pext = std::chrono::high_resolution_clock::now();
	auto ms_batch_pext = std::chrono::duration_cast<std::chrono::milliseconds>(end_batch_pext - start_batch_pext).count();

	for (size_t i = 0; i < NUM_QUERIES; ++i) {
		if (offsets_get_index[i] != offsets_zp7[i] || offsets_zp7[i] != offsets_simd[i] || offsets_simd[i] != offsets_prepare_simd[i]) {
			std::cerr << "resolve_offset mismatch at " << i
					  << ": get_index=" << static_cast<int>(offsets_get_index[i])
					  << " zp7=" << static_cast<int>(offsets_zp7[i])
					  << " simd=" << static_cast<int>(offsets_simd[i])
					  << " prepare+simd=" << static_cast<int>(offsets_prepare_simd[i])
					  << std::endl;
			return 1;
		}
	}

	std::cout << "--- resolve_offset SIMD vs zp7 ---" << std::endl;
	std::cout << "Architecture SIMD capability active: " << SIMD_ARCH_NAME << std::endl;
	std::cout << "Queries: " << NUM_QUERIES << std::endl;
	std::cout << "get_index time: " << ms_get_index << " ms" << std::endl;
	std::cout << "zp7 resolve_offset time: " << ms_zp7 << " ms" << std::endl;
	std::cout << "SIMD resolve_offset time: " << ms_simd << " ms" << std::endl;
	std::cout << "prepare+SIMD resolve_offset time: " << ms_prepare_simd << " ms" << std::endl;
	std::cout << "read_prepared time: " << ms_prepared << " ms" << std::endl;
	std::cout << "read_batch time: " << ms_batch << " ms" << std::endl;
	std::cout << "read_batch_pext time: " << ms_batch_pext << " ms" << std::endl;
	std::cout << "Speedup (read_batch / read_batch_pext): " << static_cast<double>(ms_batch) / static_cast<double>(ms_batch_pext) << "x" << std::endl;
	std::cout << "Speedup (read_prepared / read_batch_pext): " << static_cast<double>(ms_prepared) / static_cast<double>(ms_batch_pext) << "x" << std::endl;
	std::cout << "Speedup (get_index / simd): " << static_cast<double>(ms_get_index) / static_cast<double>(ms_simd) << "x" << std::endl;
	std::cout << "Speedup (get_index / prepare+simd): " << static_cast<double>(ms_get_index) / static_cast<double>(ms_prepare_simd) << "x" << std::endl;
	std::cout << "Speedup (zp7 / simd): " << static_cast<double>(ms_zp7) / static_cast<double>(ms_simd) << "x" << std::endl;
	std::cout << "Checksum (get_index/zp7/simd/prepare+simd): " << sum_get_index << " / " << sum_zp7 << " / " << sum_simd << " / " << sum_prepare_simd << std::endl;

	return 0;
}
