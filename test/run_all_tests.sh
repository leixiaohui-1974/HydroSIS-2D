#!/bin/bash

# Comprehensive test suite for HydroSIS-2D
# Tests all test cases with validation

echo "================================================"
echo "  HydroSIS-2D Comprehensive Test Suite"
echo "================================================"
echo ""

# Create output directory
mkdir -p test_results
cd test_results

# Test configuration
NX=256
NY=256
CFL=0.5
TEND=2.0

# Test cases
declare -a TEST_NAMES=(
    "1D_Dam_Break"
    "2D_Circular_Dam"
    "Partial_Dam_Break"
    "Thacker_Beach"
    "MacDonald"
    "Lake_at_Rest"
    "Small_Perturbation"
    "Flow_Over_Bump"
    "Oblique_Jump"
)

# Check if executable exists
if [ ! -f "../build/hydrosis" ]; then
    echo "Error: hydrosis executable not found!"
    echo "Please run 'cd build && make' first"
    exit 1
fi

# Run each test case
for i in {0..8}; do
    TEST_NAME="${TEST_NAMES[$i]}"
    echo ""
    echo "========================================"
    echo "Test $((i+1))/9: $TEST_NAME"
    echo "========================================"

    # Create directory for this test
    mkdir -p $TEST_NAME
    cd $TEST_NAME

    # Run with validation
    echo "Running test case $i..."
    ../../build/hydrosis \
        --test $i \
        --nx $NX \
        --ny $NY \
        --cfl $CFL \
        --tend $TEND \
        --validate \
        --vtk \
        > log.txt 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Test PASSED"

        # Extract performance metrics
        PERF=$(grep "Performance:" log.txt | awk '{print $2}')
        STEPS=$(grep "Time steps:" log.txt | awk '{print $3}')

        echo "  Performance: $PERF gigacells/s"
        echo "  Time steps: $STEPS"

        # Check for warnings
        WARNINGS=$(grep -i "warning" log.txt | wc -l)
        if [ $WARNINGS -gt 0 ]; then
            echo "  ⚠ $WARNINGS warnings detected"
        fi
    else
        echo "✗ Test FAILED (exit code: $EXIT_CODE)"
        echo "  See $TEST_NAME/log.txt for details"
    fi

    cd ..
done

echo ""
echo "================================================"
echo "  Test Suite Complete"
echo "================================================"
echo ""
echo "Results saved in test_results/"
echo ""

# Generate summary report
echo "Generating summary report..."
cat > summary_report.txt << EOF
HydroSIS-2D Test Suite Summary
================================
Date: $(date)
Grid Size: ${NX} x ${NY}
CFL: ${CFL}
End Time: ${TEND} s

Test Results:
-------------
EOF

for i in {0..8}; do
    TEST_NAME="${TEST_NAMES[$i]}"
    if [ -f "$TEST_NAME/log.txt" ]; then
        PERF=$(grep "Performance:" $TEST_NAME/log.txt | awk '{print $2}')
        STATUS=$(grep -q "Simulation completed" $TEST_NAME/log.txt && echo "PASS" || echo "FAIL")
        echo "$((i+1)). $TEST_NAME: $STATUS ($PERF gigacells/s)" >> summary_report.txt
    fi
done

echo "" >> summary_report.txt
echo "For detailed results, check individual test directories." >> summary_report.txt

cat summary_report.txt

echo ""
echo "Full report saved to test_results/summary_report.txt"
