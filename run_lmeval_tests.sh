#!/bin/bash

# LM-Eval Test Runner
# Runs all lmeval tests on a single GPU with timing instrumentation
# Usage: ./run_lmeval_tests.sh [options]
#   -c CONFIG_DIR  : Config directory (default: tests/lmeval/configs)
#   -g GPU_ID      : GPU to use (default: 0)
#   -d CADENCE     : Test cadence (default: weekly)
#   -o OUTPUT_FILE : Output log file (default: lmeval_tests_output.log)
#   -n             : Disable cache (run fresh evaluations)
#   -t TIMINGS_DIR : Timings output directory (default: timings/lm-eval)

set -e  # Exit on error (disabled for test failures)

# Default values
CONFIG_DIR="tests/lmeval/configs"
GPU_ID=0
CADENCE="weekly"
OUTPUT_FILE="lmeval_tests_output.log"
DISABLE_CACHE=0
TIMINGS_DIR="timings/lm-eval"

# Parse command line arguments
while getopts "c:g:d:o:nt:h" OPT; do
  case ${OPT} in
    c )
        CONFIG_DIR="$OPTARG"
        ;;
    g )
        GPU_ID="$OPTARG"
        ;;
    d )
        CADENCE="$OPTARG"
        ;;
    o )
        OUTPUT_FILE="$OPTARG"
        ;;
    n )
        DISABLE_CACHE=1
        ;;
    t )
        TIMINGS_DIR="$OPTARG"
        ;;
    h )
        echo "Usage: $0 [options]"
        echo "  -c CONFIG_DIR  : Config directory (default: tests/lmeval/configs)"
        echo "  -g GPU_ID      : GPU to use (default: 0)"
        echo "  -d CADENCE     : Test cadence: commit|weekly|nightly (default: weekly)"
        echo "  -o OUTPUT_FILE : Output log file (default: lmeval_tests_output.log)"
        echo "  -n             : Disable cache (run fresh evaluations)"
        echo "  -t TIMINGS_DIR : Timings output directory (default: timings/lm-eval)"
        echo "  -h             : Show this help message"
        exit 0
        ;;
    \? )
        echo "Invalid option: -$OPTARG" >&2
        echo "Use -h for help"
        exit 1
        ;;
  esac
done

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Set cadence
export CADENCE=$CADENCE

# Set timings directory
export TIMINGS_DIR=$TIMINGS_DIR

# Set cache disable if requested
if [[ $DISABLE_CACHE -eq 1 ]]; then
    export DISABLE_LMEVAL_CACHE=1
    echo "Cache DISABLED - will run fresh evaluations"
else
    echo "Cache ENABLED - will reuse base model evaluations if available"
fi

# Track overall success
SUCCESS=0
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Create timings directory
mkdir -p "$TIMINGS_DIR"

# Print configuration
echo "========================================"
echo "LM-Eval Test Runner"
echo "========================================"
echo "Config directory: $CONFIG_DIR"
echo "GPU ID: $GPU_ID"
echo "Cadence: $CADENCE"
echo "Output file: $OUTPUT_FILE"
echo "Timings directory: $TIMINGS_DIR"
echo "========================================"
echo ""

# Function to run tests and capture output
run_tests() {
    # Find all yaml config files
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo "ERROR: Config directory '$CONFIG_DIR' not found!"
        exit 1
    fi

    CONFIG_FILES=("$CONFIG_DIR"/*.yaml)

    if [[ ! -e "${CONFIG_FILES[0]}" ]]; then
        echo "ERROR: No .yaml files found in '$CONFIG_DIR'!"
        exit 1
    fi

    echo "Found ${#CONFIG_FILES[@]} config file(s)"
    echo ""

    # Iterate through all config files
    for MODEL_CONFIG in "${CONFIG_FILES[@]}"
    do
        LOCAL_SUCCESS=0
        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        CONFIG_NAME=$(basename "$MODEL_CONFIG")

        echo "========================================"
        echo "TEST $TOTAL_TESTS: $CONFIG_NAME"
        echo "========================================"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        # Set the config file for this test (use full path)
        export TEST_DATA_FILE="$MODEL_CONFIG"

        # Run pytest with capture disabled for real-time output
        if pytest \
            tests/lmeval/test_lmeval.py \
            -v \
            --capture=no \
            --tb=short; then
            LOCAL_SUCCESS=0
            PASSED_TESTS=$((PASSED_TESTS + 1))
            echo ""
            echo "✓ PASSED: $CONFIG_NAME"
        else
            LOCAL_SUCCESS=$?
            FAILED_TESTS=$((FAILED_TESTS + 1))
            echo ""
            echo "✗ FAILED: $CONFIG_NAME (exit code: $LOCAL_SUCCESS)"
        fi

        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        # Accumulate failures but continue
        SUCCESS=$((SUCCESS + LOCAL_SUCCESS))
    done

    echo "========================================"
    echo "TEST SUMMARY"
    echo "========================================"
    echo "Total tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "========================================"
    echo ""

    if [[ -d "$TIMINGS_DIR" ]]; then
        echo "Timing results saved to: $TIMINGS_DIR"
        ls -lh "$TIMINGS_DIR"/*.csv 2>/dev/null || echo "No timing files generated"
    fi

    echo ""
    echo "Full output saved to: $OUTPUT_FILE"

    if [[ $SUCCESS -eq 0 ]]; then
        echo ""
        echo "✓ ALL TESTS PASSED"
        return 0
    else
        echo ""
        echo "✗ SOME TESTS FAILED (total failures: $SUCCESS)"
        return $SUCCESS
    fi
}

# Run tests and pipe to both console and file
{
    run_tests
} 2>&1 | tee "$OUTPUT_FILE"

# Exit with the test status
exit ${PIPESTATUS[0]}
