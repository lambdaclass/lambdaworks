use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_sumcheck::{evaluate_product_at_point, prove, verify};
use std::time::Instant;

// Mersenne prime 2^61 - 1
const MODULUS: u64 = (1u64 << 61) - 1;
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

fn parse_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn generate_poly(num_vars: usize, seed: u64) -> DenseMultilinearPolynomial<F> {
    let len = 1usize << num_vars;
    let evals: Vec<FE> = (0..len)
        .map(|i| {
            let v = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64);
            FE::from(v % MODULUS)
        })
        .collect();
    DenseMultilinearPolynomial::new(evals)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_vars = parse_arg(&args, "--num-vars", 14);
    let degree = parse_arg(&args, "--degree", 2);
    let iterations = parse_arg(&args, "--iterations", 1);

    eprintln!(
        "Sumcheck bench: num_vars={}, degree={}, iterations={}",
        num_vars, degree, iterations
    );
    eprintln!(
        "  table size: 2^{} = {} entries per factor, {} factors",
        num_vars,
        1usize << num_vars,
        degree
    );

    let factors: Vec<DenseMultilinearPolynomial<F>> = (0..degree)
        .map(|i| generate_poly(num_vars, 42 + i as u64))
        .collect();

    for iter in 0..iterations {
        // Clone outside timing to isolate prove/verify costs
        let prove_factors = factors.clone();
        let verify_factors = factors.clone();

        let start = Instant::now();
        let (claimed_sum, proof_polys) = prove(prove_factors).expect("prove failed");
        let prove_time = start.elapsed();

        let start = Instant::now();
        let ok = verify(num_vars, claimed_sum, proof_polys, verify_factors)
            .expect("verify returned error");
        let verify_time = start.elapsed();

        // Measure oracle evaluation alone (verifier's bottleneck)
        let random_point: Vec<FE> = (0..num_vars).map(|i| FE::from(i as u64 + 7)).collect();
        let start = Instant::now();
        let _ = evaluate_product_at_point(&factors, &random_point);
        let oracle_eval_time = start.elapsed();

        assert!(ok, "verification failed on iteration {}", iter);

        eprintln!(
            "  iter {}: prove={:.3}ms  verify={:.3}ms  oracle_eval={:.3}ms  ratio={:.1}x",
            iter,
            prove_time.as_secs_f64() * 1000.0,
            verify_time.as_secs_f64() * 1000.0,
            oracle_eval_time.as_secs_f64() * 1000.0,
            prove_time.as_secs_f64() / verify_time.as_secs_f64(),
        );
    }
}
