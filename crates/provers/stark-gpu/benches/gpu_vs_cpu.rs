//! Benchmark: GPU STARK prover vs CPU STARK prover.
//!
//! Run with: cargo bench -p lambdaworks-stark-gpu --features metal

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
};
use stark_platinum_prover::{
    examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_stark_gpu::metal::prover::prove_gpu;

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

fn bench_proving(c: &mut Criterion) {
    let mut group = c.benchmark_group("stark_prover");
    group.sample_size(10); // STARK proving is expensive, 10 samples is enough

    // Trace lengths to benchmark: 2^10, 2^12, 2^14, 2^16
    for log_trace_len in [10, 12, 14, 16] {
        let trace_length: usize = 1 << log_trace_len;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();

        // CPU benchmark
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("2^{log_trace_len}")),
            &trace_length,
            |b, &trace_len| {
                b.iter(|| {
                    let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_len);
                    let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
                    let mut transcript = DefaultTranscript::<F>::new(&[]);
                    Prover::<F, F, _>::prove(&air, &mut trace, &mut transcript).unwrap();
                });
            },
        );

        // GPU benchmark
        #[cfg(all(target_os = "macos", feature = "metal"))]
        group.bench_with_input(
            BenchmarkId::new("gpu", format!("2^{log_trace_len}")),
            &trace_length,
            |b, &trace_len| {
                b.iter(|| {
                    let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_len);
                    let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
                    let mut transcript = DefaultTranscript::<F>::new(&[]);
                    prove_gpu(&air, &mut trace, &mut transcript).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_proving);
criterion_main!(benches);
