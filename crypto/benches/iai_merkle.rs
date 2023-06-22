use core::hint::black_box;
use lambdaworks_crypto::{
    hash::sha3::Sha3Hasher,
    merkle_tree::merkle::{Sha3_256Backend, MerkleTree},
};
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

type F = Stark252PrimeField;
type FE = FieldElement<Stark252PrimeField>;

#[inline(never)]
#[export_name = "util::build_unhashed_leaves"]
fn build_unhashed_leaves() -> Vec<FE> {
    // NOTE: the values to hash don't really matter, so let's go with the easy ones.
    core::iter::successors(Some(FE::zero()), |s| Some(s + FE::one()))
        // `(1 << 20) + 1` exploits worst cases in terms of rounding up to powers of 2.
        .take((1 << 20) + 1)
        .collect()
}

#[inline(never)]
#[export_name = "util::build_hasher"]
fn build_hasher() -> Box<Sha3Hasher> {
    Box::new(Sha3Hasher::new())
}

#[inline(never)]
fn merkle_tree_build_benchmark() {
    let unhashed_leaves = build_unhashed_leaves();
    let result = black_box(MerkleTree::<Sha3_256Backend<F>>::build(black_box(
        &unhashed_leaves,
    )));
    // Let's not count `drop` in our timings.
    core::mem::drop(result);
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*,core::mem::drop";
    functions = merkle_tree_build_benchmark,
);
