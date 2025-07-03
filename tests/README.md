# Tests

This directory contains test files for the project following Python testing conventions.

## Structure

- `test_dataloader.py` - Tests for the dataloader classes
- `__init__.py` - Makes this directory a Python package

## Running Tests

### Individual Test
```bash
python3 tests/test_dataloader.py
```

### With pytest (recommended)
```bash
pip install pytest
pytest tests/
```

## Dependencies

Tests require the same dependencies as the main project:
- torch
- tokenizers
- numpy

Install with:
```bash
pip install torch tokenizers numpy
```