use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_fft::polynomial::evaluate_fft_cpu;
use lambdaworks_fft::roots_of_unity::get_twiddles as get_twiddles_lambdaworks;
use lambdaworks_math::field::element::FieldElement as LambdaFieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsPrimeField};
use lambdaworks_math::field::traits::{IsField, RootsConfig};
use rand::Rng;
use winter_math::fft::{evaluate_poly, get_twiddles};
use winter_math::{fields::f64::BaseElement, FieldElement, StarkField};


pub fn run_winterfell(poly: &[BaseElement]) {
    let mut p = poly.to_vec();
    let twiddles = get_twiddles::<BaseElement>(p.len());
    evaluate_poly(&mut p, &twiddles);
}

pub fn run_lambdaworks(
    poly: &[LambdaFieldElement<WinterfellFieldWrapper>],
) -> Vec<LambdaFieldElement<WinterfellFieldWrapper>> {
    let p = poly.clone();
    evaluate_fft_cpu(p).unwrap()
}

pub fn run_winterfell_twiddles(poly: &[BaseElement]) {
    let p = poly.to_vec();
    get_twiddles::<BaseElement>(p.len());
}

pub fn run_lambdaworks_twiddles(poly: &[LambdaFieldElement<WinterfellFieldWrapper>]) {
    let p = poly.clone();
    let order = p.len().trailing_zeros();
    get_twiddles_lambdaworks::<WinterfellFieldWrapper>(order.into(), RootsConfig::BitReverse).unwrap();
}

#[derive(Clone, Debug)]
pub struct WinterfellFieldWrapper;

impl IsFFTField for WinterfellFieldWrapper {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: Self::BaseType =
        Self::BaseType::new(7277203076849721926u64);
}

impl IsPrimeField for WinterfellFieldWrapper {
    type RepresentativeType = u64;

    fn representative(a: &Self::BaseType) -> Self::RepresentativeType {
        a.as_int()
    }

    fn field_bit_size() -> usize {
        todo!()
    }
}

impl IsField for WinterfellFieldWrapper {
    type BaseType = BaseElement;

    fn add(a: &BaseElement, b: &BaseElement) -> BaseElement {
        a.clone() + b.clone()
    }

    fn sub(a: &BaseElement, b: &BaseElement) -> BaseElement {
        a.clone() - b.clone()
    }

    fn neg(a: &BaseElement) -> BaseElement {
        -a.clone()
    }

    fn mul(a: &BaseElement, b: &BaseElement) -> BaseElement {
        a.clone() * b.clone()
    }

    fn div(a: &BaseElement, b: &BaseElement) -> BaseElement {
        a.clone() / b.clone()
    }

    fn inv(a: &BaseElement) -> BaseElement {
        a.inv()
    }

    fn eq(a: &BaseElement, b: &BaseElement) -> bool {
        a == b
    }

    fn zero() -> BaseElement {
        BaseElement::from(0u64)
    }

    fn one() -> BaseElement {
        BaseElement::from(1u64)
    }

    fn from_u64(x: u64) -> BaseElement {
        BaseElement::from(x)
    }

    fn from_base_type(x: BaseElement) -> BaseElement {
        x
    }
}

fn fft_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");
    let mut rng = rand::thread_rng();
    let mut random_u64s: Vec<u64> = Vec::with_capacity(8192);
    for _ in 0..8192 {
        random_u64s.push(rng.gen());
    }
    let poly_winterfell: Vec<_> = random_u64s.iter().map(|x| BaseElement::from(*x)).collect();
    let poly_lambda: Vec<_> = random_u64s
        .iter()
        .map(|x| LambdaFieldElement::<WinterfellFieldWrapper>::from(*x))
        .collect();
    group.bench_function("winterfell_evaluate_fft", |bench| {
        bench.iter(|| black_box(run_winterfell(black_box(&poly_winterfell))))
    });
    group.bench_function("lambda_evaluate_fft", |bench| {
        bench.iter(|| black_box(run_lambdaworks(black_box(&poly_lambda))))
    });
    group.bench_function("winterfell_compute_twiddles", |bench| {
        bench.iter(|| black_box(run_winterfell_twiddles(black_box(&poly_winterfell))))
    });
    group.bench_function("lambda_compute_twiddles", |bench| {
        bench.iter(|| black_box(run_lambdaworks_twiddles(black_box(&poly_lambda))))
    });
}

criterion_group!(benches, fft_benches);
criterion_main!(benches);
