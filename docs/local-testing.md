# Local Testing Guide

## Professional Methods to Replicate CI/CD Locally

### 1. **Pre-commit Hooks** (Automatic - Recommended)
Runs checks automatically before every commit.

```bash
# Install and setup (already done)
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

### 2. **Makefile** (Industry Standard)
Simple, cross-platform commands for common tasks.

```bash
# Windows (if you have make installed)
make all        # Run all CI checks
make test       # Run tests only
make lint       # Run linting only
make format     # Auto-format code
make clean      # Clean cache files

# Without make, use the commands directly from Makefile
```

### 3. **Tox** (Python-specific)
Creates isolated environments to test against multiple Python versions.

```bash
# Install tox
pip install tox

# Run all environments
tox

# Run specific environment
tox -e lint     # Linting only
tox -e type     # Type checking only
tox -e py311    # Tests on Python 3.11
tox -e ci       # Full CI suite
```

### 4. **Act** (Exact GitHub Actions Replication)
Runs your actual GitHub Actions workflow locally using Docker.

```bash
# Install on Windows
winget install nektos.act

# Install on Mac
brew install act

# Install on Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run the CI workflow
act push                 # Simulate a push event
act -j test             # Run specific job
act -l                  # List all actions
act --container-architecture linux/amd64  # For M1 Macs
```

### 5. **Nox** (Alternative to Tox)
More Pythonic alternative to tox.

```python
# noxfile.py
import nox

@nox.session(python=["3.9", "3.10", "3.11"])
def tests(session):
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/")

@nox.session
def lint(session):
    session.install("black", "isort", "flake8")
    session.run("isort", "--check-only", "src/", "tests/")
    session.run("black", "--check", "src/", "tests/")
    session.run("flake8", "src/", "tests/")
```

## Recommended Workflow

### For Individual Developers:
1. **Pre-commit hooks** for automatic checking
2. **Makefile** for manual commands
3. **Act** before pushing to verify CI will pass

### For Teams:
1. **Pre-commit hooks** (enforced via `.pre-commit-config.yaml`)
2. **Tox/Nox** for testing multiple Python versions
3. **Makefile** for standardized commands
4. **Act** in CI/CD pipeline testing

## Quick Commands Comparison

| Task | Pre-commit | Make | Tox | Act |
|------|------------|------|-----|-----|
| Run all checks | `pre-commit run --all-files` | `make all` | `tox -e ci` | `act push` |
| Run tests | `pytest tests/` | `make test` | `tox` | `act -j test` |
| Format code | `pre-commit run black --all-files` | `make format` | `tox -e format` | N/A |
| Check linting | `pre-commit run --all-files` | `make lint` | `tox -e lint` | `act -j lint` |

## Why Not Shell Scripts?

Shell scripts are **not** standard practice because:
1. **Platform-specific** (bash vs PowerShell vs cmd)
2. **No dependency management**
3. **No environment isolation**
4. **Not standardized** across projects
5. **Can't test multiple Python versions**

## Best Practice Hierarchy

1. **Best**: `act` + `pre-commit` + `Makefile`
   - Exact CI replication + automatic checks + simple commands

2. **Good**: `tox` + `pre-commit` + `Makefile`
   - Multi-version testing + automatic checks + simple commands

3. **Acceptable**: `pre-commit` + `Makefile`
   - Automatic checks + simple commands

4. **Avoid**: Shell scripts only
   - Platform-specific, no standards
