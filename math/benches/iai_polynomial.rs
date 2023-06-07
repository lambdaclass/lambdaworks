use const_random::const_random;
use core::hint::black_box;
use util::FE;

mod util;

const ORDER: u64 = const_random!(u64) % 8;

#[inline(never)]
fn poly_evaluate_benchmarks() {
    let poly = util::rand_poly(ORDER);
    let x = FE::new(rand::random::<u64>());
    black_box(poly.evaluate(black_box(&x)));
}

#[inline(never)]
fn poly_evaluate_slice_benchmarks() {
    let poly = util::rand_poly(ORDER);
    let inputs = util::rand_field_elements(ORDER);
    black_box(poly.evaluate_slice(black_box(&inputs)));
}

#[inline(never)]
fn poly_add_benchmarks() {
    let x_poly = util::rand_poly(ORDER);
    let y_poly = util::rand_poly(ORDER);
    let _ = black_box(black_box(black_box(&x_poly) + black_box(&y_poly)));
}

#[inline(never)]
fn poly_neg_benchmarks() {
    let x_poly = util::rand_poly(ORDER);
    let _ = black_box(-black_box(x_poly));
}

#[inline(never)]
fn poly_sub_benchmarks() {
    let x_poly = util::rand_poly(ORDER);
    let y_poly = util::rand_poly(ORDER);
    let _ = black_box(black_box(x_poly) - black_box(y_poly));
}

#[inline(never)]
fn poly_mul_benchmarks() {
    let x_poly = util::rand_poly(ORDER);
    let y_poly = util::rand_poly(ORDER);
    let _ = black_box(black_box(x_poly) * black_box(y_poly));
}

#[inline(never)]
fn poly_div_benchmarks() {
    let x_poly = util::rand_poly(ORDER);
    let y_poly = util::rand_poly(ORDER);
    let _ = black_box(black_box(x_poly) / black_box(y_poly));
}

#[inline(never)]
fn poly_div_ruffini_benchmarks() {
    let mut x_poly = util::rand_poly(ORDER);
    let b = util::rand_field_elements(1)[0];
    black_box(&mut x_poly).ruffini_division_inplace(black_box(&b));
    let _ = black_box(x_poly);
}

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = poly_evaluate_benchmarks,
    poly_evaluate_slice_benchmarks,
    poly_add_benchmarks,
    poly_neg_benchmarks,
    poly_sub_benchmarks,
    poly_mul_benchmarks,
    poly_div_benchmarks,
    poly_div_ruffini_benchmarks,
);
