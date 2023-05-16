use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use functions::stark::{
    generate_cairo_trace, generate_fib_2_cols_proof_params, generate_fib_proof_params,
};
use lambdaworks_stark::{
    air::{context::ProofOptions, example::cairo},
    prover::prove,
};

mod functions;
mod util;

pub fn proof_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK");
    group.sample_size(10);

    for &trace_length in [64, 128, 256, 512].iter() {
        let (trace, fibonacci_air) = generate_fib_proof_params(trace_length);
        group.throughput(Throughput::Elements(trace_length as u64));
        group.bench_function(BenchmarkId::new("Simple Fibonacci", trace_length), |b| {
            b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
        });
    }

    for &trace_length in [64, 128, 256, 512].iter() {
        let (trace, fibonacci_air) = generate_fib_2_cols_proof_params(trace_length);
        group.throughput(Throughput::Elements(trace_length as u64));
        group.bench_function(BenchmarkId::new("2 column Fibonacci", trace_length), |b| {
            b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
        });
    }

    // TODO: this benchmarks panic
    // let (trace, fibonacci_air) = generate_fib17_proof_params(4);
    // group.bench_function("Fibonacci F17", |b| {
    //     b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
    // });

    // TODO: this benchmarks panic
    // let (trace, fibonacci_air) = generate_quadratic_proof_params(16);
    // group.bench_function("Quadratic AIR", |b| {
    //     b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
    // });

    run_fibonacci_bench(
        &mut group,
        "Cairo Fibonacci proof generation - 5 elements",
        "fibonacci_5",
    );
    run_fibonacci_bench(
        &mut group,
        "Cairo Fibonacci proof generation - 10 elements",
        "fibonacci_10",
    );
    run_fibonacci_bench(
        &mut group,
        "Cairo Fibonacci proof generation - 30 elements",
        "fibonacci_30",
    );
    run_fibonacci_bench(
        &mut group,
        "Cairo Fibonacci proof generation - 50 elements",
        "fibonacci_50",
    );
    run_fibonacci_bench(
        &mut group,
        "Cairo Fibonacci proof generation - 100 elements",
        "fibonacci_100",
    );
}

fn run_fibonacci_bench(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, file: &str) {
    let trace = generate_cairo_trace(file, "all_cairo");

    let proof_options = ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 5,
        coset_offset: 3,
    };
    let cairo_air = cairo::CairoAIR::new(proof_options, &trace.0);

    group.bench_function(name, |bench| {
        bench.iter(|| black_box(prove(black_box(&trace), black_box(&cairo_air))));
    });
}

criterion_group!(benches, proof_benchmarks);
criterion_main!(benches);
