//! Sumcheck benchmark example for use with hyperfine
//!
//! Usage:
//!   cargo build --release -p lambdaworks-sumcheck --example benchmark
//!   hyperfine './target/release/examples/benchmark prover-linear 12'
//!   hyperfine './target/release/examples/benchmark prover-optimized 12'
//!   hyperfine './target/release/examples/benchmark verifier 12'

use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_prime_field::U64PrimeField},
    polynomial::dense_multilinear_poly::DenseMultilinearPolynomial,
};
use lambdaworks_sumcheck::{
    prove, prove_batched, prove_blendy, prove_fast, prove_memory_efficient, prove_optimized,
    prove_parallel, prove_small_field, verify,
};
use std::env;
use std::time::Instant;

const MODULUS: u64 = 0xFFFFFFFF00000001; // Goldilocks prime
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

fn rand_field_element(seed: u64) -> FE {
    // Simple LCG for reproducible random numbers
    FE::from(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
}

fn rand_dense_multilinear_poly(num_vars: usize, seed: u64) -> DenseMultilinearPolynomial<F> {
    let size = 1 << num_vars;
    let evals: Vec<FE> = (0..size)
        .map(|i| rand_field_element(seed.wrapping_add(i as u64)))
        .collect();
    DenseMultilinearPolynomial::new(evals)
}

fn bench_prover_linear(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove(vec![poly]).unwrap();
}

fn bench_prover_quadratic(num_vars: usize) {
    let poly1 = rand_dense_multilinear_poly(num_vars, 42);
    let poly2 = rand_dense_multilinear_poly(num_vars, 123);
    let _ = prove(vec![poly1, poly2]).unwrap();
}

fn bench_prover_optimized_linear(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove_optimized(vec![poly]).unwrap();
}

fn bench_prover_optimized_quadratic(num_vars: usize) {
    let poly1 = rand_dense_multilinear_poly(num_vars, 42);
    let poly2 = rand_dense_multilinear_poly(num_vars, 123);
    let _ = prove_optimized(vec![poly1, poly2]).unwrap();
}

fn bench_prover_parallel_linear(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove_parallel(vec![poly]).unwrap();
}

fn bench_prover_fast_linear(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove_fast(vec![poly]).unwrap();
}

fn bench_prover_fast_quadratic(num_vars: usize) {
    let poly1 = rand_dense_multilinear_poly(num_vars, 42);
    let poly2 = rand_dense_multilinear_poly(num_vars, 123);
    let _ = prove_fast(vec![poly1, poly2]).unwrap();
}

fn bench_verifier_linear(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let (claimed_sum, proof_polys) = prove(vec![poly.clone()]).unwrap();
    let _ = verify(num_vars, claimed_sum, proof_polys, vec![poly]).unwrap();
}

fn bench_multilinear_evaluate(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let point: Vec<FE> = (0..num_vars)
        .map(|i| rand_field_element(100 + i as u64))
        .collect();
    let _ = poly.evaluate(point).unwrap();
}

fn bench_multilinear_fix_variable(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let r = rand_field_element(999);
    let _ = poly.fix_first_variable(&r);
}

fn bench_prover_small_field(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove_small_field(poly).unwrap();
}

fn bench_prover_blendy(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove_blendy(poly, 2).unwrap();
}

fn bench_prover_memory_efficient(num_vars: usize) {
    let poly = rand_dense_multilinear_poly(num_vars, 42);
    let _ = prove_memory_efficient(poly).unwrap();
}

fn bench_prover_batched(num_vars: usize, num_instances: usize) {
    let instances: Vec<Vec<DenseMultilinearPolynomial<F>>> = (0..num_instances)
        .map(|i| vec![rand_dense_multilinear_poly(num_vars, 42 + i as u64 * 100)])
        .collect();
    let _ = prove_batched(instances).unwrap();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <benchmark> <num_vars>", args[0]);
        eprintln!();
        eprintln!("Benchmarks:");
        eprintln!("  prover-linear      - Prove single polynomial sumcheck (original)");
        eprintln!("  prover-quadratic   - Prove two polynomial product sumcheck (original)");
        eprintln!("  prover-opt-linear  - Prove single polynomial sumcheck (optimized)");
        eprintln!("  prover-opt-quad    - Prove two polynomial product sumcheck (optimized)");
        eprintln!("  prover-parallel    - Prove with rayon parallelization");
        eprintln!("  prover-fast        - Prove with precomputed differences");
        eprintln!("  prover-small-field - Prove with small field optimizations");
        eprintln!("  prover-blendy      - Prove with memory-efficient Blendy algorithm");
        eprintln!("  prover-memory      - Prove with sqrt(N) memory usage");
        eprintln!("  prover-batched     - Prove 4 batched instances");
        eprintln!("  prover-batched-8   - Prove 8 batched instances");
        eprintln!("  verifier           - Verify sumcheck proof");
        eprintln!("  multilinear-eval   - Evaluate multilinear polynomial at point");
        eprintln!("  multilinear-fix    - Fix first variable of multilinear polynomial");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  hyperfine './target/release/examples/benchmark prover-linear 12' \\");
        eprintln!("            './target/release/examples/benchmark prover-opt-linear 12'");
        std::process::exit(1);
    }

    let benchmark = &args[1];
    let num_vars: usize = args[2].parse().expect("num_vars must be a number");

    match benchmark.as_str() {
        "prover-linear" => bench_prover_linear(num_vars),
        "prover-quadratic" => bench_prover_quadratic(num_vars),
        "prover-opt-linear" => bench_prover_optimized_linear(num_vars),
        "prover-opt-quad" => bench_prover_optimized_quadratic(num_vars),
        "prover-parallel" => bench_prover_parallel_linear(num_vars),
        "prover-fast" => bench_prover_fast_linear(num_vars),
        "prover-fast-quad" => bench_prover_fast_quadratic(num_vars),
        "prover-small-field" => bench_prover_small_field(num_vars),
        "prover-blendy" => bench_prover_blendy(num_vars),
        "prover-memory" => bench_prover_memory_efficient(num_vars),
        "prover-batched" => bench_prover_batched(num_vars, 4),
        "prover-batched-8" => bench_prover_batched(num_vars, 8),
        "verifier" => bench_verifier_linear(num_vars),
        "multilinear-eval" => bench_multilinear_evaluate(num_vars),
        "multilinear-fix" => bench_multilinear_fix_variable(num_vars),
        "all" => {
            println!(
                "Running all benchmarks with {} variables (2^{} = {} evals)...",
                num_vars,
                num_vars,
                1u64 << num_vars
            );
            println!();

            let start = Instant::now();
            bench_prover_optimized_linear(num_vars);
            println!("prover-opt-linear:  {:?}", start.elapsed());

            let start = Instant::now();
            bench_prover_optimized_quadratic(num_vars);
            println!("prover-opt-quad:    {:?}", start.elapsed());

            let start = Instant::now();
            bench_prover_linear(num_vars);
            println!("prover-linear:      {:?} (original)", start.elapsed());

            let start = Instant::now();
            bench_prover_quadratic(num_vars);
            println!("prover-quadratic:   {:?} (original)", start.elapsed());

            let start = Instant::now();
            bench_verifier_linear(num_vars);
            println!("verifier:           {:?}", start.elapsed());

            let start = Instant::now();
            bench_multilinear_evaluate(num_vars);
            println!("multilinear-eval:   {:?}", start.elapsed());

            let start = Instant::now();
            bench_multilinear_fix_variable(num_vars);
            println!("multilinear-fix:    {:?}", start.elapsed());
        }
        "compare" => {
            println!(
                "Comparing provers ({} variables, 2^{} = {} evals)...",
                num_vars,
                num_vars,
                1u64 << num_vars
            );
            println!();

            let start = Instant::now();
            bench_prover_fast_linear(num_vars);
            let fast_time = start.elapsed();
            println!("Fast prover:       {:?}", fast_time);

            let start = Instant::now();
            bench_prover_parallel_linear(num_vars);
            let parallel_time = start.elapsed();
            println!("Parallel prover:   {:?}", parallel_time);

            let start = Instant::now();
            bench_prover_optimized_linear(num_vars);
            let opt_time = start.elapsed();
            println!("Optimized prover:  {:?}", opt_time);

            if num_vars <= 12 {
                let start = Instant::now();
                bench_prover_linear(num_vars);
                let orig_time = start.elapsed();
                println!("Original prover:   {:?}", orig_time);

                println!();
                println!("Speedup vs original:");
                println!(
                    "  Fast:       {:.1}x",
                    orig_time.as_secs_f64() / fast_time.as_secs_f64()
                );
                println!(
                    "  Parallel:   {:.1}x",
                    orig_time.as_secs_f64() / parallel_time.as_secs_f64()
                );
                println!(
                    "  Optimized:  {:.1}x",
                    orig_time.as_secs_f64() / opt_time.as_secs_f64()
                );
            }
        }
        _ => {
            eprintln!("Unknown benchmark: {}", benchmark);
            std::process::exit(1);
        }
    }
}
