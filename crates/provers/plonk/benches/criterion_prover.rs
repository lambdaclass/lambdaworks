use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        default_types::FrField, pairing::BLS12381AtePairing,
    },
    field::element::FieldElement,
};
use lambdaworks_plonk::{
    prover::Prover,
    setup::setup,
    test_utils::{
        circuit_large::{test_common_preprocessed_input_size, test_witness_size},
        utils::{test_srs, TestRandomFieldGenerator},
    },
};

type Kzg = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

fn bench_plonk_prover(c: &mut Criterion) {
    let mut group = c.benchmark_group("plonk_prove");

    for n in [4096, 8192, 16384] {
        let cpi = test_common_preprocessed_input_size(n);
        let x = FieldElement::from(4_u64);
        let e = FieldElement::from(3_u64);
        let y = &x * &e;
        let witness = test_witness_size(x.clone(), e, n);
        let public_input = vec![x, y];

        let srs = test_srs(n);
        let kzg = Kzg::new(srs);
        let vk = setup(&cpi, &kzg);

        let random_generator = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg, random_generator);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(prover.prove(&witness, &public_input, &cpi, &vk));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_plonk_prover);
criterion_main!(benches);
