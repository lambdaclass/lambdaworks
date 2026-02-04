# Performance Optimization

This guide covers various techniques for optimizing lambdaworks performance.

## Native CPU Builds

For maximum performance on your local machine, build with native CPU optimizations. This enables CPU-specific instructions like AVX2/AVX-512 that can significantly speed up cryptographic operations.

### Using Environment Variables

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Using Cargo Configuration

Add to `.cargo/config.toml` in your project:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

**Note**: Binaries built with `target-cpu=native` are optimized for your specific CPU and may not run on machines with older CPUs. Only use this for local builds, not for distributed binaries.

## Link-Time Optimization (LTO)

Lambdaworks release builds use full LTO (`lto = "fat"`) by default in the workspace configuration. This allows the compiler to optimize across crate boundaries, enabling:

- Better cross-crate inlining
- Dead code elimination across the entire dependency graph
- More aggressive constant propagation

LTO increases compile time but provides 10-20% runtime performance improvement at zero code cost.

## Parallel Features

For multi-threaded operations, enable the `parallel` feature:

```bash
cargo build --release --features parallel
```

This enables parallel MSM (multi-scalar multiplication) and other parallelizable operations using rayon.

## Benchmarking

To run benchmarks with optimal settings:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench
```

The benchmark profile already has LTO enabled for accurate measurements.

## Performance Tips

1. **Use release builds**: Debug builds are significantly slower due to lack of optimizations.

2. **Enable native CPU features**: The `-C target-cpu=native` flag enables SIMD instructions specific to your CPU.

3. **Batch operations**: When possible, batch multiple operations together to amortize overhead and enable better parallelization.

4. **Pre-compute when possible**: For repeated operations with the same parameters (e.g., same generator point), pre-compute tables.

5. **Memory allocation**: For hot paths, consider pre-allocating vectors to avoid repeated allocations.
