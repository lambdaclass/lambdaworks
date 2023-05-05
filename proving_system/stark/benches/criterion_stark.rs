use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

mod functions;
mod util;

pub fn proof_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK");
    group.sample_size(10);

    for trace_length in [8, 128, 256, 512, 1024].iter() {
        group.throughput(Throughput::Bytes(*trace_length as u64));
        group.bench_with_input(
            BenchmarkId::new("Simple Fibonacci", trace_length),
            trace_length,
            |b, &t| {
                b.iter(|| functions::stark::prove_fib(t));
            },
        );
    }

    group.bench_function("2 column Fibonacci", |bench| {
        bench.iter(functions::stark::prove_fib_2_cols);
    });

    group.bench_function("Fibonacci F17", |bench| {
        bench.iter(functions::stark::prove_fib17);
    });

    group.bench_function("Quadratic AIR", |bench| {
        bench.iter(functions::stark::prove_quadratic);
    });

    group.bench_function("Cairo Fibonacci proof generation - 5 elements", |bench| {
        bench.iter(functions::stark::prove_cairo_fibonacci_5);
    });

    group.bench_function("Cairo Fibonacci proof generation - 10 elements", |bench| {
        bench.iter(functions::stark::prove_cairo_fibonacci_10);
    });

    group.bench_function("Cairo Fibonacci proof generation - 30 elements", |bench| {
        bench.iter(functions::stark::prove_cairo_fibonacci_30);
    });

    group.bench_function("Cairo Fibonacci proof generation - 50 elements", |bench| {
        bench.iter(functions::stark::prove_cairo_fibonacci_50);
    });

    group.bench_function("Cairo Fibonacci proof generation - 100 elements", |bench| {
        bench.iter(functions::stark::prove_cairo_fibonacci_100);
    });
}

criterion_group!(benches, proof_benchmarks);
criterion_main!(benches);
