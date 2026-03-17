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
/// Variables: [1, output, x_0, x_1, ..., x_{n-1}]
///   index:    0    1      2    3          n+1
///
/// Constraint i (0..n-1): x_i * x_i = x_{i+1}
/// Constraint n-1:        x_{n-1} * x_{n-1} = output
///
/// The output (x_n = 2^(2^n)) is placed at index 1 so it matches
/// the public input slot (z = (1, public, witness...)).
fn generate_chain_circuit(n: usize) -> (R1CS<F>, Vec<FE>, Vec<FE>) {
    let num_vars = n + 2; // 1 + output + x_0 + ... + x_{n-1}
    let one = FE::one();

    // Compute the chain: x_0=2, x_{i+1} = x_i^2
    let mut chain = vec![FE::from(2u64)];
    for _ in 0..n {
        let prev = chain.last().unwrap();
        chain.push(prev * prev);
    }
    // chain = [x_0, x_1, ..., x_n] where x_n is the output

    // Build witness: z = [1, output, x_0, x_1, ..., x_{n-1}]
    let mut witness = vec![one.clone(), chain[n].clone()];
    witness.extend_from_slice(&chain[..n]);

    let mut a_entries = Vec::with_capacity(n);
    let mut b_entries = Vec::with_capacity(n);
    let mut c_entries = Vec::with_capacity(n);

    for i in 0..n {
        // A and B select x_i (at column i+2 in the witness)
        a_entries.push(SparseEntry {
            row: i,
            col: i + 2,
            val: one.clone(),
        });
        b_entries.push(SparseEntry {
            row: i,
            col: i + 2,
            val: one.clone(),
        });
        // C selects the result: x_{i+1} if i < n-1, or output (col 1) if i == n-1
        let result_col = if i < n - 1 { i + 3 } else { 1 };
        c_entries.push(SparseEntry {
            row: i,
            col: result_col,
            val: one.clone(),
        });
    }

    let a = SparseMatrix::new(a_entries, n, num_vars).unwrap();
    let b = SparseMatrix::new(b_entries, n, num_vars).unwrap();
    let c = SparseMatrix::new(c_entries, n, num_vars).unwrap();

    let r1cs = R1CS::new(a, b, c, 1).unwrap();
    let public_inputs = vec![chain[n].clone()];

    (r1cs, witness, public_inputs)
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
