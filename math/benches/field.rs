use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64FieldElement;

fn u64_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("u64");

    group.bench_function("add", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        let y: U64FieldElement<13> = FieldElement::new(5);
        bench.iter(|| black_box(x) + black_box(y));
    });

    group.bench_function("mul", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        let y: U64FieldElement<13> = FieldElement::new(5);
        bench.iter(|| black_box(x) * black_box(y));
    });

    group.bench_function("pow", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        let y: u64 = 5;
        bench.iter(|| black_box(x).pow(black_box(y)));
    });

    group.bench_function("sub", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        let y: U64FieldElement<13> = FieldElement::new(5);
        bench.iter(|| black_box(x) - black_box(y));
    });

    // group.bench_function("neg", |bench| {
    // });

    group.bench_function("inv", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        bench.iter(|| black_box(x).inv());
    });

    group.bench_function("div", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        let y: U64FieldElement<13> = FieldElement::new(5);
        bench.iter(|| black_box(x) / black_box(y));
    });

    group.bench_function("eq", |bench| {
        let x: U64FieldElement<13> = FieldElement::new(2);
        let y: U64FieldElement<13> = FieldElement::new(5);
        bench.iter(|| black_box(x) == black_box(y));
    });

    // group.bench_function("from_u64", |bench| {
    // });

    // group.bench_function("from_base_type", |bench| {
    // });
}

criterion_group!(benches, u64_benchmark);
criterion_main!(benches);
