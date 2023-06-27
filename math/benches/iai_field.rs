use criterion::black_box;
use utils::u64_utils;

mod utils;

#[inline(never)]
fn fp_add_benchmarks() {
    let (x, y) = u64_utils::get_field_elements();
    let _ = black_box(black_box(x) + black_box(y));
}

#[inline(never)]
fn fp_mul_benchmarks() {
    let (x, y) = u64_utils::get_field_elements();
    let _ = black_box(black_box(x) * black_box(y));
}

#[inline(never)]
fn fp_pow_benchmarks() {
    let (x, _) = u64_utils::get_field_elements();
    let y: u64 = 5;
    let _ = black_box(black_box(x).pow(black_box(y)));
}

#[inline(never)]
fn fp_sub_benchmarks() {
    let (x, y) = u64_utils::get_field_elements();
    let _ = black_box(black_box(x) - black_box(y));
}

#[inline(never)]
fn fp_inv_benchmarks() {
    let (x, _) = u64_utils::get_field_elements();
    let _ = black_box(black_box(x).inv());
}

#[inline(never)]
fn fp_div_benchmarks() {
    let (x, y) = u64_utils::get_field_elements();
    let _ = black_box(black_box(x) / black_box(y));
}

#[inline(never)]
fn fp_eq_benchmarks() {
    let (x, y) = u64_utils::get_field_elements();
    let _ = black_box(black_box(x) == black_box(y));
}

#[inline(never)]
fn fp_sqrt_benchmarks() {
    // Make sure it has a square root
    let x = u64_utils::get_squared_field_element();
    let _ = black_box(black_box(x).sqrt());
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = fp_add_benchmarks,
    fp_mul_benchmarks,
    fp_pow_benchmarks,
    fp_sub_benchmarks,
    fp_inv_benchmarks,
    fp_div_benchmarks,
    fp_eq_benchmarks,
    fp_sqrt_benchmarks,
);
