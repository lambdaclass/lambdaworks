use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};
use lambdaworks_spartan::{
    spartan_prove, spartan_verify, SparseEntry, SparseMatrix, TrivialPCS, R1CS,
};

const MODULUS: u64 = 4611686018427387903; // 2^62 - 1 (Mersenne prime)
type F = U64PrimeField<MODULUS>;
type FE = FieldElement<F>;

/// Generates a multiplication-chain circuit with `n` constraints.
///
/// Variables: [1, x_0, x_1, ..., x_{n}]
/// Constraint i: x_i * x_i = x_{i+1}  (repeated squaring from x_0=2)
fn generate_chain_circuit(n: usize) -> (R1CS<F>, Vec<FE>, Vec<FE>) {
    let num_vars = n + 2; // 1 + x_0 + ... + x_n
    let one = FE::one();

    // Compute the chain: x_0=2, x_{i+1} = x_i^2
    let mut values = vec![one.clone()]; // z[0] = 1 (constant)
    let mut cur = FE::from(2u64);
    values.push(cur.clone());
    for _ in 0..n {
        cur = &cur * &cur;
        values.push(cur.clone());
    }

    let mut a_entries = Vec::with_capacity(n);
    let mut b_entries = Vec::with_capacity(n);
    let mut c_entries = Vec::with_capacity(n);

    for i in 0..n {
        a_entries.push(SparseEntry {
            row: i,
            col: i + 1,
            val: one.clone(),
        });
        b_entries.push(SparseEntry {
            row: i,
            col: i + 1,
            val: one.clone(),
        });
        c_entries.push(SparseEntry {
            row: i,
            col: i + 2,
            val: one.clone(),
        });
    }

    let a = SparseMatrix::new(a_entries, n, num_vars).unwrap();
    let b = SparseMatrix::new(b_entries, n, num_vars).unwrap();
    let c = SparseMatrix::new(c_entries, n, num_vars).unwrap();

    // Public input: just the final result x_n
    let r1cs = R1CS::new(a, b, c, 1).unwrap();
    let public_inputs = vec![values[n + 1].clone()];

    (r1cs, values, public_inputs)
}

fn bench_spartan(c: &mut Criterion) {
    let mut group = c.benchmark_group("spartan_trivial_pcs");

    for log_n in [4, 6, 8, 10] {
        let n = 1usize << log_n;
        let (r1cs, witness, public_inputs) = generate_chain_circuit(n);

        group.bench_with_input(
            BenchmarkId::new("prove", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    spartan_prove(&r1cs, &public_inputs, &witness, TrivialPCS).unwrap();
                });
            },
        );

        // Pre-generate proof for verify benchmark
        let proof = spartan_prove(&r1cs, &public_inputs, &witness, TrivialPCS).unwrap();

        group.bench_with_input(
            BenchmarkId::new("verify", format!("2^{log_n}")),
            &n,
            |b, _| {
                b.iter(|| {
                    spartan_verify(&r1cs, &public_inputs, &proof, TrivialPCS).unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_spartan);
criterion_main!(benches);
