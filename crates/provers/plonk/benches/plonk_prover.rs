//! PLONK prover and verifier benchmarks.
//!
//! Run with: cargo bench -p lambdaworks-plonk

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{
    FrElement, FrField,
};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::traits::IsRandomFieldElementGenerator;
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_plonk::prover::Prover;
use lambdaworks_plonk::setup::{setup, CommonPreprocessedInput};
use lambdaworks_plonk::verifier::Verifier;

type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;
type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked("7");

/// Deterministic RNG for benchmarks (no blinding for consistent results).
#[derive(Clone)]
struct BenchRandomGenerator;
impl IsRandomFieldElementGenerator<FrField> for BenchRandomGenerator {
    fn generate(&self) -> FrElement {
        FrElement::zero()
    }
}

/// Generates a test SRS for benchmarking.
fn bench_srs(n: usize) -> StructuredReferenceString<G1Point, G2Point> {
    let s = FrElement::from(2);
    let g1 = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2 = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..n + 3)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
}

/// Creates a simple circuit with n multiplication gates.
/// Circuit: x_0 * x_1 = x_2, x_2 * x_3 = x_4, ...
fn create_mul_chain_circuit(num_constraints: usize) -> ConstraintSystem<FrField> {
    let mut cs = ConstraintSystem::new();

    // Create initial inputs
    let x = cs.new_public_input();
    let y = cs.new_variable();

    // First multiplication
    let mut prev = cs.mul(&x, &y);

    // Chain more multiplications to reach desired constraint count
    for _ in 1..num_constraints {
        let next = cs.new_variable();
        prev = cs.mul(&prev, &next);
    }

    // Add a public output constraint
    let output = cs.new_public_input();
    cs.assert_eq(&prev, &output);

    cs
}

/// Creates witness values for the multiplication chain circuit.
fn create_mul_chain_witness(
    num_constraints: usize,
) -> (lambdaworks_plonk::setup::Witness<FrField>, Vec<FrElement>) {
    // For simplicity, use values that multiply to give predictable results
    // x=2, all multipliers=1, so output=2
    let x = FrElement::from(2u64);
    let output = FrElement::from(2u64);

    // Build witness arrays
    // This is a simplified witness - in practice you'd solve the constraints
    let n = (num_constraints + 1).next_power_of_two();

    let mut a = vec![FrElement::zero(); n];
    let mut b = vec![FrElement::zero(); n];
    let mut c = vec![FrElement::zero(); n];

    // First constraint: x * 1 = x
    a[0] = x.clone();
    b[0] = FrElement::one();
    c[0] = x.clone();

    // Remaining constraints: prev * 1 = prev
    for i in 1..num_constraints {
        a[i] = x.clone();
        b[i] = FrElement::one();
        c[i] = x.clone();
    }

    // Padding constraint for assertion
    a[num_constraints] = output.clone();
    b[num_constraints] = FrElement::one();
    c[num_constraints] = output.clone();

    let witness = lambdaworks_plonk::setup::Witness { a, b, c };
    let public_input = vec![x, output];

    (witness, public_input)
}

fn benchmark_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("PLONK Prover");
    group.sample_size(10); // Reduce sample size for large circuits

    // Benchmark different circuit sizes (powers of 2)
    for size in [64, 256, 1024, 4096, 16384, 32768] {
        let cs = create_mul_chain_circuit(size);
        let cpi = CommonPreprocessedInput::from_constraint_system(&cs, &ORDER_R_MINUS_1_ROOT_UNITY)
            .expect("Failed to create CPI");

        let srs = bench_srs(cpi.n);
        let kzg = KZG::new(srs);
        let vk = setup(&cpi, &kzg);

        let (witness, public_input) = create_mul_chain_witness(size);
        let prover = Prover::new(kzg.clone(), BenchRandomGenerator);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("{}_constraints", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        prover
                            .prove(
                                black_box(&witness),
                                black_box(&public_input),
                                black_box(&cpi),
                                black_box(&vk),
                            )
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

fn benchmark_verifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("PLONK Verifier");
    group.sample_size(10);

    // Benchmark different circuit sizes
    for size in [64, 256, 1024, 4096, 16384, 32768] {
        let cs = create_mul_chain_circuit(size);
        let cpi = CommonPreprocessedInput::from_constraint_system(&cs, &ORDER_R_MINUS_1_ROOT_UNITY)
            .expect("Failed to create CPI");

        let srs = bench_srs(cpi.n);
        let kzg = KZG::new(srs);
        let vk = setup(&cpi, &kzg);

        let (witness, public_input) = create_mul_chain_witness(size);
        let prover = Prover::new(kzg.clone(), BenchRandomGenerator);

        // Create proof once
        let proof = prover
            .prove(&witness, &public_input, &cpi, &vk)
            .expect("Failed to create proof");

        let verifier = Verifier::new(kzg);

        group.bench_with_input(
            BenchmarkId::new("verify", format!("{}_constraints", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(verifier.verify(
                        black_box(&proof),
                        black_box(&public_input),
                        black_box(&cpi),
                        black_box(&vk),
                    ))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("PLONK Setup");
    group.sample_size(10);

    for size in [64, 256, 1024, 4096, 16384, 32768] {
        let cs = create_mul_chain_circuit(size);
        let cpi = CommonPreprocessedInput::from_constraint_system(&cs, &ORDER_R_MINUS_1_ROOT_UNITY)
            .expect("Failed to create CPI");

        let srs = bench_srs(cpi.n);
        let kzg = KZG::new(srs);

        group.bench_with_input(
            BenchmarkId::new("setup", format!("{}_constraints", size)),
            &size,
            |b, _| {
                b.iter(|| black_box(setup(black_box(&cpi), black_box(&kzg))));
            },
        );
    }

    group.finish();
}

fn benchmark_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("PLONK End-to-End");
    group.sample_size(10);

    for size in [256, 1024, 4096] {
        group.bench_with_input(
            BenchmarkId::new("prove_and_verify", format!("{}_constraints", size)),
            &size,
            |b, &size| {
                b.iter(|| {
                    // Build circuit
                    let cs = create_mul_chain_circuit(size);
                    let cpi = CommonPreprocessedInput::from_constraint_system(
                        &cs,
                        &ORDER_R_MINUS_1_ROOT_UNITY,
                    )
                    .expect("Failed to create CPI");

                    // Setup
                    let srs = bench_srs(cpi.n);
                    let kzg = KZG::new(srs);
                    let vk = setup(&cpi, &kzg);

                    // Prove
                    let (witness, public_input) = create_mul_chain_witness(size);
                    let prover = Prover::new(kzg.clone(), BenchRandomGenerator);
                    let proof = prover
                        .prove(&witness, &public_input, &cpi, &vk)
                        .expect("Failed to prove");

                    // Verify
                    let verifier = Verifier::new(kzg);
                    black_box(verifier.verify(&proof, &public_input, &cpi, &vk))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_prover,
    benchmark_verifier,
    benchmark_setup,
    benchmark_end_to_end
);
criterion_main!(benches);
