# Python bindings

This library provides a Python FFI. It can be imported with `import lambdaworks_py`. Currently, it supports FieldElement over U256 operations, with operators.

To run tests, set up a virtual env using python 3.11.2:
```
python3 -m venv .venv
source .venv/bin/activate
make py.develop
make test
```
