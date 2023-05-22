use const_random::const_random;
use iai_callgrind::black_box;
use util::{rand_field_elements, rand_poly, FE};

mod util;

const ORDER: u64 = const_random!(u64) % 8;

#[inline(never)]
fn poly_evaluate_benchmarks() {
    let poly = rand_poly(ORDER);
    let x = FE::new(rand::random::<u64>());
    poly.evaluate(black_box(&x));
}

#[inline(never)]
fn poly_evaluate_slice_benchmarks() {
    let poly = rand_poly(ORDER);
    let inputs = rand_field_elements(ORDER);
    poly.evaluate_slice(black_box(&inputs));
}

#[inline(never)]
fn poly_add_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(&x_poly) + black_box(&y_poly);
}

#[inline(never)]
fn poly_neg_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let _ = black_box(x_poly);
}

#[inline(never)]
fn poly_sub_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(x_poly) - black_box(y_poly);
}

#[inline(never)]
fn poly_mul_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(x_poly) + black_box(y_poly);
}

#[inline(never)]
fn poly_div_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(x_poly) + black_box(y_poly);
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = poly_evaluate_benchmarks,
    poly_evaluate_slice_benchmarks,
    poly_add_benchmarks,
    poly_neg_benchmarks,
    poly_sub_benchmarks,
    poly_mul_benchmarks,
    poly_div_benchmarks
);
