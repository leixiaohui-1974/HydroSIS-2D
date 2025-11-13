# Contributing to HydroSIS-2D

Thank you for your interest in contributing to HydroSIS-2D! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or experience level.

### Expected Behavior

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or personal attacks
- Trolling, insulting comments, or inflammatory language
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## Getting Started

### Prerequisites

1. **Development Environment**:
   - NVIDIA GPU with CUDA support (for GPU development)
   - CUDA Toolkit 11.0+ (12.0+ recommended)
   - Python 3.10+
   - CMake 3.18+
   - Git

2. **Knowledge Requirements**:
   - Python programming
   - CUDA/GPU programming (for solver contributions)
   - Shallow water equations (for numerical methods)
   - Version control with Git

### Setting Up Development Environment

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/HydroSIS-2D.git
cd HydroSIS-2D

# 3. Add upstream remote
git remote add upstream https://github.com/leixiaohui-1974/HydroSIS-2D.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install development dependencies
pip install -r requirements-dev.txt

# 6. Build GPU solver (if working on solver)
cd src/solver
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
make install

# 7. Run tests to verify setup
cd ../../../
pytest prepost/tests/ -v
```

---

## Development Workflow

### Branch Strategy

We use a feature branch workflow:

```
main (stable releases)
  ↑
develop (integration branch)
  ↑
feature/your-feature-name (your work)
```

### Creating a Feature Branch

```bash
# Update your local repository
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Example branch names:
# - feature/urban-infiltration
# - feature/netcdf-io
# - bugfix/mass-conservation
# - docs/api-reference
```

### Commit Messages

Use clear, descriptive commit messages following this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Example**:
```
feat(solver): Add infiltration source term

Implement Green-Ampt infiltration model for urban flood modeling.
Includes CUDA kernel implementation and Python bindings.

Closes #42
```

### Keeping Your Branch Updated

```bash
# Regularly sync with upstream
git fetch upstream
git rebase upstream/develop

# Resolve conflicts if any
# Then continue with:
git rebase --continue
```

---

## Coding Standards

### Python Code

Follow **PEP 8** style guide:

```bash
# Format code with Black
black prepost/ examples/

# Check style with flake8
flake8 prepost/ --max-line-length=100

# Sort imports with isort
isort prepost/ examples/
```

**Key Guidelines**:
- Line length: 100 characters (relaxed from PEP 8's 79)
- Indentation: 4 spaces
- Use type hints for function signatures
- Docstrings: Google style

**Example**:
```python
def compute_flux(h_left: float, h_right: float, u_left: float, u_right: float,
                 g: float = 9.81) -> tuple[float, float, float]:
    """
    Compute HLL flux for shallow water equations.

    Args:
        h_left: Water depth at left state (m)
        h_right: Water depth at right state (m)
        u_left: Velocity at left state (m/s)
        u_right: Velocity at right state (m/s)
        g: Gravitational acceleration (m/s²)

    Returns:
        Tuple of (F_h, F_hu, F_hv) fluxes
    """
    # Implementation
    pass
```

### CUDA Code

Follow **CUDA best practices**:

```cpp
// File header comment
/**
 * @file flux_kernels.cu
 * @brief CUDA kernels for flux computation
 * @author HydroSIS-2D Contributors
 * @date 2025-11-13
 */

// Clear kernel documentation
/**
 * @brief Compute HLL flux in x-direction
 *
 * @param h Water depth array
 * @param u Velocity x-component
 * @param flux_x Output flux array
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Grid spacing in x
 * @param g Gravitational acceleration
 */
__global__ void computeFluxX_HLL(
    const double* __restrict__ h,
    const double* __restrict__ u,
    double* __restrict__ flux_x,
    int nx, int ny,
    double dx, double g
)
{
    // Implementation
}
```

**Guidelines**:
- Use `__restrict__` for pointer parameters
- Thread/block indexing at top of kernel
- Clear variable names (no single letters except indices)
- Comment complex algorithms

### C++ Code

Follow **C++17 standard**:

```cpp
class ShallowWaterSolver {
public:
    /// Initialize solver with configuration
    void initialize(const SolverConfig& config);

    /// Run simulation to specified end time
    void run(double t_end);

private:
    // Member variables
    SolverConfig config_;
    DeviceMemory dev_mem_;

    // Helper methods
    void allocateMemory();
    void freeMemory();
};
```

---

## Testing Guidelines

### Test Requirements

**All contributions must include tests**. No pull request will be merged without adequate test coverage.

### Test Categories

1. **Unit Tests** (Fast, focused)
   - Test individual functions/methods
   - No GPU required
   - Location: `prepost/tests/test_*.py`

2. **Integration Tests** (Module interactions)
   - Test component interactions
   - May require GPU
   - Location: `prepost/tests/test_integration.py`

3. **Validation Tests** (Accuracy verification)
   - Compare against analytical solutions
   - Requires GPU
   - Location: `prepost/tests/validation/`

4. **Performance Tests** (Regression detection)
   - Ensure no performance degradation
   - Requires GPU
   - Location: `prepost/tests/performance/`

### Writing Tests

Use **pytest** framework:

```python
import pytest
import numpy as np
from preprocessing.mesh_generation import MeshGenerator, DomainParams


class TestMeshGeneration:
    """Test mesh generation functionality"""

    def test_uniform_mesh_creation(self):
        """Test basic uniform mesh creation"""
        # Arrange
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
        mesh_gen = MeshGenerator(domain)

        # Act
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=50)

        # Assert
        assert mesh.nx == 100
        assert mesh.ny == 50
        assert mesh.ncells == 5000
        assert mesh.dx == pytest.approx(1.0, rel=1e-10)

    def test_mesh_boundary_identification(self):
        """Test boundary cell identification"""
        # Test implementation
        pass

    @pytest.mark.gpu
    def test_gpu_mesh_allocation(self):
        """Test GPU memory allocation for mesh"""
        # Only runs if GPU available
        pass
```

### Running Tests

```bash
# All unit tests
pytest prepost/tests/ -v

# Specific test file
pytest prepost/tests/test_mesh_generation.py -v

# Specific test
pytest prepost/tests/test_mesh_generation.py::TestMeshGeneration::test_uniform_mesh_creation -v

# With coverage
pytest prepost/tests/ -v --cov=preprocessing --cov-report=html

# Only GPU tests
pytest prepost/tests/ -v -m gpu

# Skip GPU tests
pytest prepost/tests/ -v -m "not gpu"

# Complete validation suite
python tests/run_full_validation.py
```

### Test Coverage Requirements

- **New features**: ≥ 80% line coverage
- **Bug fixes**: Test reproducing the bug + fix verification
- **Refactoring**: Maintain existing coverage

---

## Documentation

### Documentation Requirements

All contributions must include appropriate documentation:

1. **Code Documentation**:
   - Docstrings for all public functions/classes
   - Inline comments for complex logic
   - Type hints for Python code

2. **User Documentation**:
   - Update relevant guides (if user-facing feature)
   - Add examples for new features
   - Update README if necessary

3. **Developer Documentation**:
   - Technical design documents for major features
   - API documentation updates
   - Architecture diagrams (if applicable)

### Docstring Format

Use **Google style** docstrings:

```python
def create_dam_break_simulation(
    length: float,
    width: float,
    nx: int,
    ny: int,
    upstream_depth: float,
    downstream_depth: float,
    dam_position: float = 0.5
) -> SimulationConfig:
    """
    Create complete configuration for dam break simulation.

    This is a convenience function that sets up a standard dam break
    scenario with specified parameters. It creates the mesh, terrain,
    initial conditions, and boundary conditions automatically.

    Args:
        length: Domain length in x-direction (meters)
        width: Domain width in y-direction (meters)
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        upstream_depth: Water depth upstream of dam (meters)
        downstream_depth: Water depth downstream of dam (meters)
        dam_position: Relative position of dam (0.0 to 1.0, default 0.5)

    Returns:
        SimulationConfig object containing mesh, IC, BC, and parameters

    Raises:
        ValueError: If parameters are invalid (negative depths, etc.)

    Example:
        >>> config = create_dam_break_simulation(
        ...     length=200.0, width=100.0,
        ...     nx=200, ny=100,
        ...     upstream_depth=10.0,
        ...     downstream_depth=1.0
        ... )
        >>> solver.run(config)

    Note:
        This creates a flat terrain. For complex terrain, use
        GeometryGenerator directly.

    See Also:
        GeometryGenerator.custom_terrain: For complex terrain setup
        InitialConditionManager: For advanced IC setup
    """
    # Implementation
    pass
```

---

## Submitting Changes

### Pre-Submission Checklist

Before submitting a pull request, ensure:

- [ ] Code follows style guidelines (run `black`, `flake8`, `isort`)
- [ ] All tests pass (`pytest prepost/tests/ -v`)
- [ ] New tests added for new features/fixes
- [ ] Documentation updated (docstrings, guides, README)
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up-to-date with `upstream/develop`
- [ ] No merge conflicts

### Creating a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**:
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Base: `leixiaohui-1974/HydroSIS-2D` `develop`
   - Compare: `YOUR_USERNAME/HydroSIS-2D` `feature/your-feature-name`

3. **PR Title**: Clear, descriptive (e.g., "Add infiltration model for urban flooding")

4. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation
   Why is this change needed?

   ## Changes
   - Added X
   - Modified Y
   - Fixed Z

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Validation tests pass (if applicable)
   - [ ] Manual testing performed

   ## Documentation
   - [ ] Docstrings updated
   - [ ] User guide updated (if needed)
   - [ ] Examples added (if needed)

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] All tests pass
   - [ ] Documentation complete
   - [ ] Branch up-to-date

   ## Related Issues
   Closes #123
   Related to #456
   ```

5. **Request Review**: Tag maintainers for review

---

## Review Process

### What to Expect

1. **Initial Review** (1-3 days):
   - Maintainer will review code, tests, docs
   - Feedback provided as comments
   - May request changes

2. **Revision Cycle**:
   - Address feedback
   - Push updates to same branch
   - PR automatically updates
   - Request re-review

3. **Approval** (when all feedback addressed):
   - Maintainer approves PR
   - CI/CD checks pass
   - PR merged into `develop`

### Review Criteria

Your PR will be evaluated on:

- **Correctness**: Does it work as intended?
- **Testing**: Adequate test coverage?
- **Code Quality**: Follows standards, well-structured?
- **Documentation**: Clear, complete?
- **Performance**: No regressions?
- **Maintainability**: Easy to understand and modify?

### Responding to Feedback

- **Be respectful**: Reviewers are trying to help
- **Ask questions**: If feedback unclear, ask for clarification
- **Explain decisions**: If you disagree, explain your reasoning
- **Iterate quickly**: Respond to feedback promptly

---

## Contribution Types

### Bug Fixes

1. **Report the bug** (if not already reported):
   - Create GitHub issue with reproduction steps
   - Include error messages, expected vs actual behavior

2. **Write a test** that reproduces the bug:
   ```python
   def test_mass_conservation_bug():
       """Test that reproduces mass conservation issue #123"""
       # Setup that triggers bug
       # Assert incorrect behavior
   ```

3. **Fix the bug**

4. **Verify test now passes**

5. **Submit PR** referencing issue

### New Features

1. **Discuss first**: Create issue to discuss feature before implementing

2. **Design**: Write design document for major features

3. **Implement**:
   - Core functionality
   - Tests (aim for ≥80% coverage)
   - Documentation
   - Example (if user-facing)

4. **Submit PR** with complete implementation

### Documentation

Documentation improvements are always welcome:

- Fixing typos
- Clarifying explanations
- Adding examples
- Improving API docs
- Translating to other languages

Small doc fixes can be submitted directly. Larger doc changes should be discussed first.

### Performance Improvements

1. **Benchmark first**: Establish baseline performance

2. **Profile**: Identify bottlenecks

3. **Optimize**: Implement improvements

4. **Benchmark again**: Measure improvement

5. **Submit PR** with before/after metrics

---

## Development Tips

### CUDA Development

**Debugging CUDA Code**:
```bash
# Compile with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-g -G"

# Use cuda-gdb
cuda-gdb python
(cuda-gdb) run examples/01_basic_dam_break.py

# Use cuda-memcheck
cuda-memcheck python examples/01_basic_dam_break.py
```

**Profiling**:
```bash
# Nsight Systems (timeline)
nsys profile -o profile python examples/01_basic_dam_break.py

# Nsight Compute (kernel details)
ncu -o kernel_profile python examples/01_basic_dam_break.py
```

### Python Development

**Interactive Development**:
```python
# Use IPython for interactive testing
ipython

from preprocessing.mesh_generation import MeshGenerator, DomainParams
domain = DomainParams(0, 100, 0, 50)
mesh_gen = MeshGenerator(domain)
mesh = mesh_gen.generate_uniform_mesh(100, 50)
```

**Debugging**:
```python
# Use pdb
import pdb; pdb.set_trace()

# Or ipdb (better interface)
import ipdb; ipdb.set_trace()
```

### Testing Tips

**Run specific tests during development**:
```bash
# Run only tests matching pattern
pytest -k "test_mesh" -v

# Run failed tests only
pytest --lf

# Run with stdout (see print statements)
pytest -s

# Stop at first failure
pytest -x
```

---

## Getting Help

### Resources

- **Documentation**: See [docs/](docs/) for comprehensive guides
- **Examples**: Check [examples/](examples/) for usage patterns
- **Tests**: Look at existing tests for examples

### Asking Questions

- **GitHub Discussions**: For general questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Code Comments**: Ask questions in PR reviews

### Communication

- Be specific: Include code snippets, error messages, steps to reproduce
- Be patient: Maintainers are volunteers
- Be respectful: We're all learning

---

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Mention of significant contributions
- **Git History**: Your commits are part of project history

---

## License

By contributing to HydroSIS-2D, you agree that your contributions will be licensed under the same license as the project (to be determined).

---

## Questions?

If you have questions about contributing, please:

1. Check this guide first
2. Search existing issues/discussions
3. Ask in GitHub Discussions
4. Contact maintainers

---

**Thank you for contributing to HydroSIS-2D!** 🌊

Your contributions help make high-performance flood modeling accessible to everyone.

---

*Last updated: 2025-11-13*
