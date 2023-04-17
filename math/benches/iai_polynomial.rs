use const_random::const_random;
use iai::black_box;
use util::{rand_field_elements, rand_poly, FE};

mod util;

const ORDER: u64 = const_random!(u64) % 8;

pub fn evaluate_benchmarks() {
    let poly = rand_poly(ORDER);
    let x = FE::new(rand::random::<u64>());
    poly.evaluate(black_box(&x));
}

pub fn evaluate_slice_benchmarks() {
    let poly = rand_poly(ORDER);
    let inputs = rand_field_elements(ORDER);
    poly.evaluate_slice(black_box(&inputs));
}

pub fn add_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(&x_poly) + black_box(&y_poly);
}

pub fn neg_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let _ = black_box(x_poly);
}

pub fn sub_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(x_poly) - black_box(y_poly);
}

pub fn mul_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(x_poly) + black_box(y_poly);
}

pub fn div_benchmarks() {
    let x_poly = rand_poly(ORDER);
    let y_poly = rand_poly(ORDER);
    let _ = black_box(x_poly) + black_box(y_poly);
}

iai::main!(
    evaluate_benchmarks,
    evaluate_slice_benchmarks,
    add_benchmarks,
    neg_benchmarks,
    sub_benchmarks,
    mul_benchmarks,
    div_benchmarks
);
