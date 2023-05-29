use criterion::black_box;
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

mod util;

#[inline(never)]
fn fp_add_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();
    let _ = black_box(x) + black_box(y);
}

#[inline(never)]
fn fp_mul_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();

    let _ = black_box(x) * black_box(y);
}

#[inline(never)]
fn fp_pow_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();

    let y: u64 = 5;
    let _ = black_box(x).pow(black_box(y));
}

#[inline(never)]
fn fp_sub_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();
    let _ = black_box(x) - black_box(y);
}

#[inline(never)]
fn fp_inv_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();

    let _ = black_box(x).inv();
}

#[inline(never)]
fn fp_div_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();
    let _ = black_box(x) / black_box(y);
}

#[inline(never)]
fn fp_eq_benchmarks() {
    let x = FieldElement::<Stark252PrimeField>::from_hex(
        "0x03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
    )
    .unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex(
        "0x0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
    )
    .unwrap();
    let _ = black_box(x) == black_box(y);
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = fp_add_benchmarks,
    fp_mul_benchmarks,
    fp_pow_benchmarks,
    fp_sub_benchmarks,
    fp_inv_benchmarks,
    fp_div_benchmarks,
    fp_eq_benchmarks
);
