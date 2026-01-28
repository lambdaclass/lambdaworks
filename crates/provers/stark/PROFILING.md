# STARK Prover Profiling Guide

This guide explains how to profile the STARK prover for memory usage and CPU performance.

## Quick Start

```bash
# Build the profiling binary
cargo build --release -p stark-platinum-prover --bench prover_profile

# Run with default settings (2^16 = 65536 rows)
./target/release/deps/prover_profile-*[!.d] --trace-length 16
```

## Memory Profiling with dhat

[dhat](https://github.com/nnethercote/dhat-rs) provides detailed heap profiling including:
- Total bytes allocated
- Peak memory usage (t-gmax)
- Allocation hotspots
- Memory leaks

### Running dhat

```bash
# Build with dhat-heap feature
cargo build --release -p stark-platinum-prover --bench prover_profile --features dhat-heap

# Run the profiler
./target/release/deps/prover_profile-*[!.d] --trace-length 14
```

This generates `dhat-heap.json` which can be viewed at:
https://nnethercote.github.io/dh_view/dh_view.html

### Example Output

```
dhat: Total:     200,078,545 bytes in 952,106 blocks
dhat: At t-gmax: 33,292,816 bytes in 98,327 blocks
dhat: At t-end:  64 bytes in 1 blocks
```

- **Total**: All allocations during execution
- **At t-gmax**: Peak memory usage
- **At t-end**: Memory still allocated at exit (potential leaks)

## CPU Profiling with samply

[samply](https://github.com/mstange/samply) is a sampling profiler that works well on macOS and Linux.

### Running samply

```bash
# Build without dhat (for accurate timing)
cargo build --release -p stark-platinum-prover --bench prover_profile

# Record a profile
samply record ./target/release/deps/prover_profile-*[!.d] --trace-length 16
```

This opens a web browser with an interactive flame graph viewer.

### Tips for samply

- Use larger trace lengths (16-18) for more meaningful profiles
- The `--iterations` flag can help accumulate more samples:
  ```bash
  samply record ./target/release/deps/prover_profile-*[!.d] --trace-length 14 --iterations 5
  ```

## Flamegraph Generation

[cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph) generates SVG flame graphs.

### Running flamegraph

```bash
# Generate flamegraph (may require sudo on macOS for dtrace)
cargo flamegraph --bench prover_profile -p stark-platinum-prover -- --trace-length 16

# Or with specific output file
cargo flamegraph --bench prover_profile -p stark-platinum-prover -o prover.svg -- --trace-length 16
```

### macOS Notes

On macOS, you may need to:
1. Disable SIP (System Integrity Protection) for dtrace, or
2. Run with sudo: `sudo cargo flamegraph ...`

## Parallel Profiling

To profile with parallelization enabled:

```bash
# Build with parallel feature
cargo build --release -p stark-platinum-prover --bench prover_profile --features parallel

# Profile
samply record ./target/release/deps/prover_profile-*[!.d] --trace-length 16 --parallel
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--trace-length <N>` | Trace length exponent (2^N rows) | 16 |
| `--iterations <N>` | Number of prove iterations | 1 |
| `--parallel` | Enable parallel mode | disabled |
| `--help` | Show help | - |

## Recommended Trace Lengths

| Trace Length | Rows | Use Case |
|--------------|------|----------|
| 10 | 1,024 | Quick tests |
| 14 | 16,384 | Development profiling |
| 16 | 65,536 | Standard benchmarking |
| 18 | 262,144 | Production-like workloads |
| 20 | 1,048,576 | Stress testing |

## Interpreting Results

### Memory Hotspots

Common memory-intensive operations in the STARK prover:
1. **LDE (Low-Degree Extension)**: Polynomial evaluations over larger domains
2. **Merkle Tree Construction**: Commitment phase allocations
3. **FFT Operations**: Temporary buffers for transforms
4. **FRI Protocol**: Layer commitments and decommitments

### CPU Hotspots

Common CPU-intensive operations:
1. **Field Arithmetic**: Modular multiplication and inversion
2. **FFT/IFFT**: Fast Fourier Transforms
3. **Polynomial Operations**: Evaluation, interpolation
4. **Hash Computations**: Merkle tree hashing

## Comparing Optimizations

To compare before/after an optimization:

```bash
# Create baseline profile
./target/release/deps/prover_profile-*[!.d] --trace-length 16 2>&1 | tee baseline.txt

# Apply optimization, rebuild, re-run
./target/release/deps/prover_profile-*[!.d] --trace-length 16 2>&1 | tee optimized.txt

# Compare
diff baseline.txt optimized.txt
```

For statistical comparison, use hyperfine:

```bash
hyperfine --warmup 1 --runs 5 \
  './target/release/deps/prover_profile-*[!.d] --trace-length 14'
```
