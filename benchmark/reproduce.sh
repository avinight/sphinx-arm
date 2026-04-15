#!/bin/bash

# Sphinx Paper Reproducibility Script
# Usage: ./benchmark/reproduce.sh [--stage 1-5 | --clean | --deep-clean]

# Always run from project root
cd "$(dirname "$0")/.." || exit 1

STAGE=$2
if [ "$1" == "--clean-build" ]; then STAGE="clean-build"; fi
if [ "$1" == "--clean-data" ]; then STAGE="clean-data"; fi
if [ "$1" == "--deep-clean" ]; then STAGE="deep-clean"; fi
CONFIG_FILE="config/config.h"
BUILD_DIR="build"

# Helper to toggle config defines
# Usage: set_config [DEFINE_NAME] [enable|disable]
set_config() {
    local name=$1
    local mode=$2
    if [ "$mode" == "enable" ]; then
        sed -i '' "s|// #define $name|#define $name|g" "$CONFIG_FILE"
        sed -i '' "s|//#define $name|#define $name|g" "$CONFIG_FILE"
    else
        sed -i '' "s|^#define $name|// #define $name|g" "$CONFIG_FILE"
    fi
}

# Helper to toggle constexpr bool in config.h
# Usage: set_constexpr_bool [NAME] [true|false]
set_constexpr_bool() {
    local name=$1
    local value=$2
    sed -i '' -E "s|^constexpr bool ${name} = (true|false);|constexpr bool ${name} = ${value};|g" "$CONFIG_FILE"
}

# Helper to build
do_build() {
    echo "--- Building Sphinx ---"
    /opt/homebrew/bin/cmake -B "$BUILD_DIR" .
    /opt/homebrew/bin/cmake --build "$BUILD_DIR" -j$(sysctl -n hw.ncpu)
}

# Initialize directories
mkdir -p benchmark/data-lf benchmark/data-ssd benchmark/data-optane \
         benchmark/data-ssd-zipf benchmark/data-optane-zipf \
         benchmark/data-skew benchmark/data-tail benchmark/data-memory \
         benchmark/data-memory-zipf benchmark/data-extra-bits benchmark/data-mt

case $STAGE in
    1)
        echo "=== STAGE 1: Preparation & SSD Baseline (Fig 10, 12) ==="
        # Ensure standard config
        set_config "IN_MEMORY_FILE" "disable"
        set_config "ENABLE_XDP" "disable"
        set_config "ENABLE_MT" "disable"
        set_config "ENABLE_BP_FOR_READ" "disable"
        set_constexpr_bool "MAIN_BENCHMARK_ZIPF" "false"
        
        do_build
        
        echo "--- Running benchmark_main (SSD) ---"
        ./"$BUILD_DIR"/benchmark/benchmark_main
        
        echo "--- Running benchmark_tail ---"
        ./"$BUILD_DIR"/benchmark/benchmark_tail

        echo "--- Running benchmark_main (SSD/Optane Zipf) ---"
        set_constexpr_bool "MAIN_BENCHMARK_ZIPF" "true"
        do_build
        ."/$BUILD_DIR"/benchmark/benchmark_main
        set_constexpr_bool "MAIN_BENCHMARK_ZIPF" "false"
        echo "Stage 1 Complete. Data available in data-ssd and data-tail."
        ;;
    2)
        echo "=== STAGE 2: Memory & XDP Analysis (Fig 7, 8, 9) ==="
        # Configure for Memory/XDP
        set_config "IN_MEMORY_FILE" "enable"
        set_config "ENABLE_XDP" "enable"
        set_config "ENABLE_MT" "disable"
        set_config "ENABLE_BP_FOR_READ" "disable"
        set_constexpr_bool "MAIN_BENCHMARK_ZIPF" "false"
        do_build
        
        echo "--- Running benchmark_main (In-Memory) ---"
        ./"$BUILD_DIR"/benchmark/benchmark_main
        
        echo "--- Running benchmark_xdp ---"
        ./"$BUILD_DIR"/benchmark/benchmark_xdp
        
        echo "--- Running benchmark_mixed (In-Memory Zipf) ---"
        ./"$BUILD_DIR"/benchmark/benchmark_mixed
        echo "Stage 2 Complete. Data available in data-memory and data-memory-zipf."
        ;;
    3)
        echo "=== STAGE 3: Advanced Workloads (Skew & MT) (Fig 11) ==="
        # Configure for Skew/MT
        set_config "IN_MEMORY_FILE" "disable"
        set_config "ENABLE_MT" "enable"
        set_config "ENABLE_BP_FOR_READ" "enable"
        
        do_build
        
        echo "--- Running benchmark_multi_threaded ---"
        ./"$BUILD_DIR"/benchmark/benchmark_multi_threaded
        
        echo "--- Running benchmark_mixed ---"
        ./"$BUILD_DIR"/benchmark/benchmark_mixed
        
        echo "--- Running benchmark_read_latency2 ---"
        ."/$BUILD_DIR"/benchmark/benchmark_read_latency2
        echo "Stage 3 Complete. Data available in data-mt and data-skew."
        ;;
    4)
        echo "=== STAGE 4: Structural Properties (Fig 13, 14) ==="
        # These are standard config
        set_config "IN_MEMORY_FILE" "disable"
        set_config "ENABLE_MT" "disable"
        set_config "ENABLE_XDP" "disable"
        set_config "ENABLE_BP_FOR_READ" "disable"
        
        do_build
        
        echo "--- Running benchmark_extra_bits ---"
        ./"$BUILD_DIR"/benchmark/benchmark_extra_bits
        
        echo "--- Running benchmark_load_factor_query_insert ---"
        ./"$BUILD_DIR"/benchmark/benchmark_load_factor_query_insert
        echo "Stage 4 Complete. Data available in data-extra-bits and data-lf."
        ;;
    5)
        echo "=== STAGE 5: Final Visualization ==="
        echo "--- Generating All Figures ---"
        source .venv/bin/activate
        python3 benchmark/mem_fpr.py
        python3 benchmark/main_exp_mem_ptr.py
        python3 benchmark/main_exp_mem.py
        python3 benchmark/main_exp.py
        python3 benchmark/combined.py
        python3 benchmark/tail.py
        python3 benchmark/extra_bits.py
        python3 benchmark/lf.py
        echo "Stage 5 Complete. SVGs generated in benchmark/ folder."
        ;;
    clean-build)
        echo "=== CLEANUP: Removing Build Artifacts (Object Files & CMake Cache) ==="
        rm -rf "$BUILD_DIR"
        rm -rf benchmark/build
        echo "Build Cleanup Complete."
        ;;
    clean-data)
        echo "=== CLEANUP: Removing Collected Data & SVGs ==="
        rm -rf benchmark/data-*
        rm -f benchmark/*.svg
        find . -name "__pycache__" -type d -exec rm -rf {} +
        echo "Data Cleanup Complete."
        ;;
    deep-clean)
        echo "=== DEEP CLEANUP: Removing Everything Including Large Backing Files ==="
        rm -rf "$BUILD_DIR"
        rm -rf benchmark/build
        rm -rf benchmark/data-*
        rm -f benchmark/*.svg
        rm -f benchmark/directory_*.txt
        find . -name "__pycache__" -type d -exec rm -rf {} +
        echo "Deep Cleanup Complete."
        ;;
    *)
        echo "Please specify a stage: 1, 2, 3, 4, or 5."
        exit 1
        ;;
esac
