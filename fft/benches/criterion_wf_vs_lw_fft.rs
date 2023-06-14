use rand::Rng;
use criterion::{Criterion, criterion_group, criterion_main, black_box};
use lambdaworks_fft::polynomial::evaluate_fft_cpu;
use lambdaworks_math::field::element::FieldElement as LambdaFieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::traits::{IsFFTField, IsPrimeField};
use winter_math::fft::evaluate_poly;
use winter_math::{
    fft::fft_inputs::FftInputs,
    get_power_series,
    {fields::f64::BaseElement, FieldElement, StarkField},
};
const MIN_CONCURRENT_SIZE: usize = 1024;

fn permute<E: FieldElement>(v: &mut [E]) {
    if cfg!(feature = "concurrent") && v.len() >= MIN_CONCURRENT_SIZE {
        #[cfg(feature = "concurrent")]
        concurrent::permute(v);
    } else {
        FftInputs::permute(v);
    }
}

pub fn get_twiddles<B>(domain_size: usize) -> Vec<B>
where
    B: StarkField,
{
    assert!(
        domain_size.is_power_of_two(),
        "domain size must be a power of 2"
    );
    assert!(
        domain_size.ilog2() <= B::TWO_ADICITY,
        "multiplicative subgroup of size {domain_size} does not exist in the specified base field"
    );
    let root = B::get_root_of_unity(domain_size.ilog2());
    let mut twiddles = get_power_series(root, domain_size / 2);
    permute(&mut twiddles);
    twiddles
}

pub fn run_winterfell(poly: &[BaseElement]) {
    let mut p = poly.to_vec();
    let twiddles = get_twiddles::<BaseElement>(p.len());
    evaluate_poly(&mut p, &twiddles);
}

pub fn run_lambdaworks(poly: &[LambdaFieldElement<WinterfellFieldWrapper>]) -> Vec<LambdaFieldElement<WinterfellFieldWrapper>> {
    let p = poly.clone();
    evaluate_fft_cpu(p).unwrap()
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
    let poly_lambda: Vec<_> = random_u64s.iter().map(|x| LambdaFieldElement::<WinterfellFieldWrapper>::from(*x)).collect();
    group.bench_function("winterfel_evaluate_fft", |bench| {
        bench.iter(|| {
            black_box(run_winterfell(black_box(&poly_winterfell)))
        })
    });
    group.bench_function("lambda_evaluate_fft", |bench| {
        bench.iter(|| {
            black_box(run_lambdaworks(black_box(&poly_lambda)))
        })
    });
}

criterion_group!(benches, fft_benches);
criterion_main!(benches);
