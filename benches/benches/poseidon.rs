use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::ByteConversion;
use pathfinder_crypto::MontFelt;
use rand::{RngCore, SeedableRng};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn poseidon_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon");

    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut felt1: [u8; 32] = Default::default();
    rng.fill_bytes(&mut felt1);
    let mut felt2: [u8; 32] = Default::default();
    rng.fill_bytes(&mut felt2);

    let lw_x = FieldElement::<Stark252PrimeField>::from_bytes_be(&felt1).unwrap();
    let lw_y = FieldElement::<Stark252PrimeField>::from_bytes_be(&felt2).unwrap();
    group.bench_function("lambdaworks", |bench| {
        bench.iter(|| black_box(PoseidonCairoStark252::hash(&lw_x, &lw_y)))
    });

    let mut mont_x = lw_x.value().limbs;
    let mut mont_y = lw_y.value().limbs;

    // In order use the same field elements for starknet-rs and pathfinder, we have to reverse
    // the limbs order respect to the lambdaworks implementation.
    mont_x.reverse();
    mont_y.reverse();

    let sn_ff_x = starknet_crypto::FieldElement::from_mont(mont_x);
    let sn_ff_y = starknet_crypto::FieldElement::from_mont(mont_y);
    group.bench_function("starknet-rs", |bench| {
        bench.iter(|| black_box(starknet_crypto::poseidon_hash(sn_ff_x, sn_ff_y)))
    });

    let pf_x = MontFelt(mont_x);
    let pf_y = MontFelt(mont_y);
    group.bench_function("pathfinder", |bench| {
        bench.iter(|| black_box(pathfinder_crypto::hash::poseidon_hash(pf_x, pf_y)))
    });
}
criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
