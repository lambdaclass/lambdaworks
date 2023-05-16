use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lambdaworks_stark::prover::prove;

mod functions;
mod util;

pub fn proof_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK");
    group.sample_size(10);

    let sizes = [1 << 7]; //, 512, 1024];

    // for trace_length in sizes.iter() {
    //     group.throughput(Throughput::Bytes(*trace_length as u64));
    //     group.bench_with_input(
    //         BenchmarkId::new("Simple Fibonacci", trace_length),
    //         trace_length,
    //         |b, &t| {
    //             b.iter(|| black_box(functions::stark::prove_fib(t)));
    //         },
    //     );
    // }

    // group.bench_function("2 column Fibonacci", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_fib_2_cols()));
    // });

    // group.bench_function("Fibonacci F17", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_fib17()));
    // });

    // group.bench_function("Quadratic AIR", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_quadratic()));
    // });
    let (raw_trace, memory, cairo_air) = functions::stark::generate_cairo_fibonacci_5_trace();
    let tuple = (raw_trace, memory);

    group.bench_function("Cairo Fibonacci proof generation - 5 elements", |bench| {
        bench.iter(|| black_box(prove(black_box(&tuple), black_box(&cairo_air))));
    });

    // group.bench_function("Cairo Fibonacci proof generation - 10 elements", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_cairo_fibonacci_10()));
    // });

    // group.bench_function("Cairo Fibonacci proof generation - 30 elements", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_cairo_fibonacci_30()));
    // });

    // group.bench_function("Cairo Fibonacci proof generation - 50 elements", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_cairo_fibonacci_50()));
    // });

    // group.bench_function("Cairo Fibonacci proof generation - 100 elements", |bench| {
    //     bench.iter(|| black_box(functions::stark::prove_cairo_fibonacci_100()));
    // });
}

criterion_group!(benches, proof_benchmarks);
criterion_main!(benches);
