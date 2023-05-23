use criterion::black_box;
use u64_utils::rand_field_elements_pair;

mod u64_utils;

#[inline(never)]
fn fp_add_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) + black_box(y);
}

#[inline(never)]
fn fp_mul_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) * black_box(y);
}

#[inline(never)]
fn fp_pow_benchmarks() {
    let (x, _) = rand_field_elements_pair();
    let y: u64 = 5;
    let _ = black_box(x).pow(black_box(y));
}

#[inline(never)]
fn fp_sub_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) - black_box(y);
}

#[inline(never)]
fn fp_inv_benchmarks() {
    let (x, _) = rand_field_elements_pair();
    let _ = black_box(x).inv();
}

#[inline(never)]
fn fp_div_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) / black_box(y);
}

#[inline(never)]
fn fp_eq_benchmarks() {
    let (x, y) = rand_field_elements_pair();
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
