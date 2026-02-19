//! Benchmark comparing batch vs individual GKR proving.
//!
//! Usage:
//!   cargo bench -p lambdaworks-gkr-logup --bench batch_vs_individual -- compare 14 4

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::DenseMultilinearPolynomial;
use std::env;
use std::time::Instant;

use lambdaworks_gkr_logup::{prove, prove_batch, Layer};

const MODULUS: u64 = 2013265921; // Baby Bear-like prime
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

fn rand_field_element(seed: u64) -> FE {
    FE::from(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
}

fn make_grand_product_layer(num_vars: usize, seed: u64) -> Layer<F> {
    let size = 1usize << num_vars;
    let values: Vec<FE> = (0..size)
        .map(|i| rand_field_element(seed.wrapping_add(i as u64)))
        .collect();
    Layer::GrandProduct(DenseMultilinearPolynomial::new(values))
}

fn make_logup_layer(num_vars: usize, seed: u64) -> Layer<F> {
    let size = 1usize << num_vars;
    let numerators: Vec<FE> = (0..size)
        .map(|i| rand_field_element(seed.wrapping_add(i as u64)))
        .collect();
    let denominators: Vec<FE> = (0..size)
        .map(|i| rand_field_element(seed.wrapping_add(1000 + i as u64)))
        .collect();
    Layer::LogUpGeneric {
        numerators: DenseMultilinearPolynomial::new(numerators),
        denominators: DenseMultilinearPolynomial::new(denominators),
    }
}

/// Prove N instances individually (separate prove + transcript per instance).
fn bench_individual_grand_product(num_vars: usize, n_instances: usize) {
    for i in 0..n_instances {
        let layer = make_grand_product_layer(num_vars, 42 + i as u64 * 100);
        let mut channel = DefaultTranscript::<F>::new(&[]);
        let _ = prove(&mut channel, layer);
    }
}

/// Prove N instances in a single batch.
fn bench_batch_grand_product(num_vars: usize, n_instances: usize) {
    let layers: Vec<Layer<F>> = (0..n_instances)
        .map(|i| make_grand_product_layer(num_vars, 42 + i as u64 * 100))
        .collect();
    let mut channel = DefaultTranscript::<F>::new(&[]);
    let _ = prove_batch(&mut channel, layers);
}

/// Prove N LogUp instances individually.
fn bench_individual_logup(num_vars: usize, n_instances: usize) {
    for i in 0..n_instances {
        let layer = make_logup_layer(num_vars, 42 + i as u64 * 100);
        let mut channel = DefaultTranscript::<F>::new(&[]);
        let _ = prove(&mut channel, layer);
    }
}

/// Prove N LogUp instances in a single batch.
fn bench_batch_logup(num_vars: usize, n_instances: usize) {
    let layers: Vec<Layer<F>> = (0..n_instances)
        .map(|i| make_logup_layer(num_vars, 42 + i as u64 * 100))
        .collect();
    let mut channel = DefaultTranscript::<F>::new(&[]);
    let _ = prove_batch(&mut channel, layers);
}

/// Prove N instances with mixed sizes (half at num_vars, half at num_vars-2).
fn bench_individual_mixed_sizes(num_vars: usize, n_instances: usize) {
    for i in 0..n_instances {
        let vars = if i % 2 == 0 {
            num_vars
        } else {
            num_vars.saturating_sub(2).max(1)
        };
        let layer = make_grand_product_layer(vars, 42 + i as u64 * 100);
        let mut channel = DefaultTranscript::<F>::new(&[]);
        let _ = prove(&mut channel, layer);
    }
}

/// Batch prove N instances with mixed sizes.
fn bench_batch_mixed_sizes(num_vars: usize, n_instances: usize) {
    let layers: Vec<Layer<F>> = (0..n_instances)
        .map(|i| {
            let vars = if i % 2 == 0 {
                num_vars
            } else {
                num_vars.saturating_sub(2).max(1)
            };
            make_grand_product_layer(vars, 42 + i as u64 * 100)
        })
        .collect();
    let mut channel = DefaultTranscript::<F>::new(&[]);
    let _ = prove_batch(&mut channel, layers);
}

fn print_comparison(label: &str, individual: std::time::Duration, batch: std::time::Duration) {
    let ratio = individual.as_secs_f64() / batch.as_secs_f64();
    let saving_pct = (1.0 - batch.as_secs_f64() / individual.as_secs_f64()) * 100.0;
    println!("  {label}");
    println!("    Individual: {:>10.3?}", individual);
    println!("    Batch:      {:>10.3?}", batch);
    if ratio >= 1.0 {
        println!("    Speedup:    {:.2}x ({:.1}% faster)", ratio, saving_pct);
    } else {
        println!(
            "    Slowdown:   {:.2}x ({:.1}% slower)",
            1.0 / ratio,
            -saving_pct
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: {} <mode> <num_vars> <n_instances>", args[0]);
        eprintln!();
        eprintln!("Modes:");
        eprintln!("  individual          - Prove N GrandProduct instances separately");
        eprintln!("  batch               - Prove N GrandProduct instances in a batch");
        eprintln!("  individual-logup    - Prove N LogUp instances separately");
        eprintln!("  batch-logup         - Prove N LogUp instances in a batch");
        eprintln!("  individual-mixed    - Prove N mixed-size instances separately");
        eprintln!("  batch-mixed         - Prove N mixed-size instances in a batch");
        eprintln!("  compare             - Run all modes and print comparison");
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  cargo bench -p lambdaworks-gkr-logup --bench batch_vs_individual -- compare 14 4"
        );
        std::process::exit(1);
    }

    let mode = &args[1];
    let num_vars: usize = args[2].parse().expect("num_vars must be a number");
    let n_instances: usize = args[3].parse().expect("n_instances must be a number");

    match mode.as_str() {
        "individual" => bench_individual_grand_product(num_vars, n_instances),
        "batch" => bench_batch_grand_product(num_vars, n_instances),
        "individual-logup" => bench_individual_logup(num_vars, n_instances),
        "batch-logup" => bench_batch_logup(num_vars, n_instances),
        "individual-mixed" => bench_individual_mixed_sizes(num_vars, n_instances),
        "batch-mixed" => bench_batch_mixed_sizes(num_vars, n_instances),
        "compare" => {
            println!(
                "=== GKR Batch vs Individual Benchmark ===\n\
                 Instances: {}, Variables: {} (2^{} = {} elements each)\n",
                n_instances,
                num_vars,
                num_vars,
                1u64 << num_vars
            );

            // GrandProduct
            let start = Instant::now();
            bench_individual_grand_product(num_vars, n_instances);
            let individual_gp = start.elapsed();

            let start = Instant::now();
            bench_batch_grand_product(num_vars, n_instances);
            let batch_gp = start.elapsed();

            print_comparison("GrandProduct (same size)", individual_gp, batch_gp);

            // LogUp
            let start = Instant::now();
            bench_individual_logup(num_vars, n_instances);
            let individual_logup = start.elapsed();

            let start = Instant::now();
            bench_batch_logup(num_vars, n_instances);
            let batch_logup = start.elapsed();

            print_comparison("LogUp (same size)", individual_logup, batch_logup);

            // Mixed sizes
            let start = Instant::now();
            bench_individual_mixed_sizes(num_vars, n_instances);
            let individual_mixed = start.elapsed();

            let start = Instant::now();
            bench_batch_mixed_sizes(num_vars, n_instances);
            let batch_mixed = start.elapsed();

            print_comparison(
                &format!(
                    "GrandProduct (mixed: {} and {} vars)",
                    num_vars,
                    num_vars.saturating_sub(2).max(1)
                ),
                individual_mixed,
                batch_mixed,
            );
        }
        _ => {
            eprintln!("Unknown mode: {}", mode);
            std::process::exit(1);
        }
    }
}
