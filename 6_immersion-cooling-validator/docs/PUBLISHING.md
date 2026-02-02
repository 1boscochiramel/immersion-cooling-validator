# Publishing to PyPI

## Prerequisites

1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org)
2. Create API tokens for both

## Build

```bash
pip install build twine
python -m build
```

## Test on TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install
pip install --index-url https://test.pypi.org/simple/ immersion-cooling-validator
```

## Publish to PyPI

```bash
twine upload dist/*
```

## Version Bump

1. Update version in `pyproject.toml`
2. Create git tag: `git tag v0.1.1`
3. Push: `git push origin main --tags`

## GitHub Actions (Automated)

The CI/CD workflow automatically:
- Runs tests on Python 3.9-3.11
- Builds package on main branch
- Creates artifacts for release

To enable auto-publish to PyPI:
1. Add `PYPI_API_TOKEN` secret to GitHub repo
2. Add publish step to `.github/workflows/ci.yml`
