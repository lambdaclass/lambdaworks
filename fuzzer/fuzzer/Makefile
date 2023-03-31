FFI_LIB_PATH=./gnark_backend

check: clippy test test-go

build-go:
	$ cd ${FFI_LIB_PATH}; \
		go build -buildmode=c-archive -o libgnark_backend.a main.go

build: build-go
	$ RUSTFLAGS="-L${FFI_LIB_PATH}" cargo build
