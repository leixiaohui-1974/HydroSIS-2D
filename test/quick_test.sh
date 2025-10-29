#!/bin/bash

# Quick sanity test
echo "Running quick sanity test..."

# Check if executable exists
if [ ! -f "../build/hydrosis" ]; then
    echo "Error: hydrosis executable not found!"
    echo "Please run 'cd build && make' first"
    exit 1
fi

# Run dam break test (smallest/fastest)
echo ""
echo "Test: 1D Dam Break (256x256, 1.0s)"
../build/hydrosis --test 0 --nx 256 --ny 256 --tend 1.0 --validate

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Quick test PASSED"
    echo ""
    echo "To run full test suite:"
    echo "  ./test/run_all_tests.sh"
    exit 0
else
    echo ""
    echo "✗ Quick test FAILED"
    exit 1
fi
