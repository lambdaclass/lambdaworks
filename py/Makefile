.PHONY: build.develop
build.develop:
	maturin develop

.PHONY: test
test: build.develop
	python tests/lamdaworks_py_test.py
