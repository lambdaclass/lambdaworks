use criterion::black_box;
use util::rand_field_elements_pair;

mod util;

fn add_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) + black_box(y);
}

fn mul_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) * black_box(y);
}

fn pow_benchmarks() {
    let (x, _) = rand_field_elements_pair();
    let y: u64 = 5;
    let _ = black_box(x).pow(black_box(y));
}

fn sub_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) - black_box(y);
}

fn inv_benchmarks() {
    let (x, _) = rand_field_elements_pair();
    let _ = black_box(x).inv();
}

fn div_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) / black_box(y);
}

fn eq_benchmarks() {
    let (x, y) = rand_field_elements_pair();
    let _ = black_box(x) == black_box(y);
}

iai::main!(
    add_benchmarks,
    mul_benchmarks,
    pow_benchmarks,
    sub_benchmarks,
    inv_benchmarks,
    div_benchmarks,
    eq_benchmarks
);
