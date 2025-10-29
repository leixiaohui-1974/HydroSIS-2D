#!/bin/bash

# Test script for configuration file system
# Demonstrates the new config file functionality

echo "================================================"
echo "  Testing Configuration File System"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

HYDROSIS="../build/hydrosis"

# Check if binary exists
if [ ! -f "$HYDROSIS" ]; then
    echo -e "${YELLOW}Note: hydrosis binary not found (expected - requires GPU compilation)${NC}"
    echo "This script demonstrates the usage syntax."
    echo "Once compiled on GPU system, these commands will work."
    echo ""
    HYDROSIS="hydrosis"
fi

echo "=== Test 1: Load dam break config ==="
echo "Command:"
echo "  $HYDROSIS --config ../examples/config_dam_break.ini --test 0"
echo ""

echo "=== Test 2: Load lake at rest config ==="
echo "Command:"
echo "  $HYDROSIS --config ../examples/config_lake_at_rest.ini --test 5"
echo ""

echo "=== Test 3: Load config with overrides ==="
echo "Command:"
echo "  $HYDROSIS --config ../examples/config_dam_break.ini \\"
echo "            --nx 1024 --ny 512 --cfl 0.4"
echo ""

echo "=== Test 4: River flow config (for future boundary conditions) ==="
echo "Command:"
echo "  $HYDROSIS --config ../examples/config_river_flow.ini --test 0"
echo "Note: Will use wall boundaries until Task 1.1 is complete"
echo ""

echo "=== Test 5: Multi-GPU with config ==="
echo "Command:"
echo "  mpirun -np 4 $HYDROSIS --config ../examples/config_dam_break.ini \\"
echo "                          --multi-gpu --nx 2048"
echo ""

echo "=== Test 6: Config with VTK output ==="
echo "Command:"
echo "  $HYDROSIS --config ../examples/config_dam_break.ini --vtk"
echo ""

echo "=== Test 7: Config with validation ==="
echo "Command:"
echo "  $HYDROSIS --config ../examples/config_lake_at_rest.ini \\"
echo "            --test 5 --validate"
echo ""

echo "================================================"
echo "  Configuration File System Tests Defined"
echo "================================================"
echo ""
echo -e "${GREEN}✓ All test commands prepared${NC}"
echo ""
echo "To run tests once compiled on GPU system:"
echo "  cd test && ./test_config_system.sh"
echo ""
echo "Manual test (check config loading):"
echo "  $HYDROSIS --config ../examples/config_dam_break.ini --help"
echo ""

# Validate config files exist
echo "Checking configuration files..."
CONFIG_FILES=(
    "../examples/config_dam_break.ini"
    "../examples/config_lake_at_rest.ini"
    "../examples/config_river_flow.ini"
)

ALL_EXIST=true
for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo -e "  ${GREEN}✓${NC} Found: $config"
    else
        echo -e "  ${YELLOW}✗${NC} Missing: $config"
        ALL_EXIST=false
    fi
done

echo ""
if [ "$ALL_EXIST" = true ]; then
    echo -e "${GREEN}✓ All configuration files present${NC}"
else
    echo -e "${YELLOW}! Some configuration files missing${NC}"
fi

echo ""
echo "Next: Compile on GPU system to run actual tests"
echo ""
