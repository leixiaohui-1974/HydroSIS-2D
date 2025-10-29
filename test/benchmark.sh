#!/bin/bash

# Performance benchmarking script
echo "================================================"
echo "  HydroSIS-2D Performance Benchmark"
echo "================================================"
echo ""

# Create output directory
mkdir -p benchmark_results
cd benchmark_results

# Test different grid sizes
declare -a SIZES=(128 256 512 1024 2048)

echo "Testing 1D Dam Break with varying grid sizes..."
echo ""

# Single GPU benchmark
echo "Grid Size,Cells,Time Steps,Total Time (s),Performance (gigacells/s)" > scaling_single_gpu.csv

for SIZE in "${SIZES[@]}"; do
    echo "Running ${SIZE}x${SIZE}..."

    ../build/hydrosis \
        --test 0 \
        --nx $SIZE \
        --ny $SIZE \
        --cfl 0.5 \
        --tend 1.0 \
        > log_${SIZE}.txt 2>&1

    if [ $? -eq 0 ]; then
        CELLS=$((SIZE * SIZE))
        STEPS=$(grep "Time steps:" log_${SIZE}.txt | awk '{print $3}')
        TIME=$(grep "Total wall time:" log_${SIZE}.txt | awk '{print $4}')
        PERF=$(grep "Performance:" log_${SIZE}.txt | awk '{print $2}')

        echo "${SIZE},${CELLS},${STEPS},${TIME},${PERF}" >> scaling_single_gpu.csv
        echo "  ${SIZE}x${SIZE}: $PERF gigacells/s"
    else
        echo "  ${SIZE}x${SIZE}: FAILED"
    fi
done

echo ""
echo "================================================"
echo "Benchmark complete!"
echo "Results saved to benchmark_results/scaling_single_gpu.csv"
echo "================================================"

# Print summary
echo ""
echo "Performance Summary:"
echo "-------------------"
cat scaling_single_gpu.csv | column -t -s,

cd ..
