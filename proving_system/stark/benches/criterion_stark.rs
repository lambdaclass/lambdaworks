use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion, Throughput,
};
use functions::stark::{
    generate_cairo_trace, generate_fib17_proof_params, generate_fib_2_cols_proof_params,
    generate_fib_proof_params, generate_quadratic_proof_params,
};
use lambdaworks_stark::{
    air::{context::ProofOptions, example::cairo},
    prover::prove,
};

mod functions;
mod util;

pub fn artificial_trace_proofs(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK");
    group.sample_size(10);

    for &trace_length in [64, 128, 256, 512].iter() {
        let (trace, fibonacci_air) = generate_fib_proof_params(trace_length);
        group.throughput(Throughput::Elements(trace_length as u64));
        group.bench_function(BenchmarkId::new("fibonacci/simple", trace_length), |b| {
            b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
        });
    }

    for &trace_length in [64, 128, 256, 512].iter() {
        let (trace, fibonacci_air) = generate_fib_2_cols_proof_params(trace_length);
        group.throughput(Throughput::Elements(trace_length as u64));
        group.bench_function(BenchmarkId::new("fibonacci/2cols", trace_length), |b| {
            b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
        });
    }

    let (trace, fibonacci_air) = generate_fib17_proof_params(4);
    group.bench_function("fibonacci/F17", |b| {
        b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
    });

    let (trace, fibonacci_air) = generate_quadratic_proof_params(16);
    group.bench_function("quadratic_air", |b| {
        b.iter(|| black_box(prove(black_box(&trace), black_box(&fibonacci_air))));
    });
}

fn cairo_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("CAIRO");
    group.sample_size(10);

    run_cairo_bench(&mut group, "fibonacci/5", "fibonacci_5");
    run_cairo_bench(&mut group, "fibonacci/10", "fibonacci_10");
    run_cairo_bench(&mut group, "fibonacci/30", "fibonacci_30");
    run_cairo_bench(&mut group, "fibonacci/50", "fibonacci_50");
    run_cairo_bench(&mut group, "fibonacci/100", "fibonacci_100");

    run_cairo_bench(&mut group, "factorial/8", "factorial_8");
    run_cairo_bench(&mut group, "factorial/16", "factorial_16");
}

fn run_cairo_bench(group: &mut BenchmarkGroup<'_, WallTime>, benchname: &str, file: &str) {
    let trace = generate_cairo_trace(file);

    let blowup_factors = [2, 4, 8];
    let query_numbers = [16, 32, 64];

    for &blowup_factor in blowup_factors.iter() {
        for &fri_number_of_queries in query_numbers.iter() {
            let proof_options = ProofOptions {
                blowup_factor,
                fri_number_of_queries,
                coset_offset: 3,
            };
            let cairo_air = cairo::CairoAIR::new(proof_options, &trace.0);

            let name = format!("{benchname}_b{blowup_factor}_q{fri_number_of_queries})");

            group.bench_function(name, |bench| {
                bench.iter(|| black_box(prove(black_box(&trace), black_box(&cairo_air))));
            });
        }
    }
}

criterion_group!(benches, artificial_trace_proofs, cairo_benches);
criterion_main!(benches);
