use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_math::cyclic_group::IsCyclicBilinearGroup;
use lambdaworks_math::elliptic_curve::traits::HasDistortionMap;
use lambdaworks_math::msm::naive::Naive;
use lambdaworks_math::{
    elliptic_curve::{
        curves::test_curve::{QuadraticNonResidue, ORDER_P},
        element::EllipticCurveElement,
        traits::HasEllipticCurveOperations,
    },
    field::{
        element::FieldElement, fields::u64_prime_field::U64PrimeField,
        quadratic_extension::QuadraticExtensionField,
    },
    msm::{pippenger, MSM},
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type FE = FieldElement<U64PrimeField<ORDER_P>>;

macro_rules! bench_msm_for_length {
    ( $func:ident, $a:expr ) => {
        pub fn $func(c: &mut Criterion) {
            let msm_size = $a;
            let window_sizes = vec![1, 2, 3, 4];

            let mut rng = StdRng::seed_from_u64(42);
            let g = EllipticCurveElement::<CurrentCurve>::generator();

            // Generate Field Elements and Group elements.
            let s: Vec<FE> = (0..msm_size).map(|_| FE::new(rng.gen())).collect();
            let hiding: Vec<EllipticCurveElement<CurrentCurve>> = (0..msm_size)
                .map(|x| g.operate_with_self(x as u128))
                .collect();

            let mut group = c.benchmark_group(format!("msm_{}", $a));

            let (s, h) = (&s[..msm_size], &hiding[..msm_size]);
            // Bench naive implementation.
            group.bench_function(BenchmarkId::new("naive", msm_size), |b| {
                let msm = Naive {};
                b.iter(|| {
                    msm.msm(s, h);
                })
            });

            // Bench Pippenger with different window sizes.
            for window_size in window_sizes.iter() {
                group.bench_with_input(
                    BenchmarkId::new("pippenger", format!("({}, {})", msm_size, *window_size)),
                    window_size,
                    |b, window_size| {
                        let msm = pippenger::Pippenger::new(*window_size);
                        b.iter(|| {
                            msm.msm(s, h);
                        })
                    },
                );
            }

            group.finish();
        }
    };
}

bench_msm_for_length!(msm_length_1, 1);
bench_msm_for_length!(msm_length_10, 10);
bench_msm_for_length!(msm_length_100, 100);
bench_msm_for_length!(msm_length_1000, 1000);

criterion_group!(
    benches,
    msm_length_1,
    msm_length_10,
    msm_length_100,
    msm_length_1000
);

criterion_main!(benches);

#[derive(Clone, Debug)]
pub struct CurrentCurve;
impl HasEllipticCurveOperations for CurrentCurve {
    type BaseField = QuadraticExtensionField<U64PrimeField<ORDER_P>, QuadraticNonResidue>;

    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(1)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn generator_affine_x() -> FieldElement<Self::BaseField> {
        FieldElement::from(35)
    }

    fn generator_affine_y() -> FieldElement<Self::BaseField> {
        FieldElement::from(31)
    }

    fn embedding_degree() -> u32 {
        2
    }

    fn order_r() -> u64 {
        5
    }

    fn order_p() -> u64 {
        59
    }
}

impl HasDistortionMap for CurrentCurve {
    fn distorsion_map(
        p: &[FieldElement<Self::BaseField>; 3],
    ) -> [FieldElement<Self::BaseField>; 3] {
        let (x, y, z) = (&p[0], &p[1], &p[2]);
        let t = FieldElement::new([FieldElement::zero(), FieldElement::one()]);
        [-x, y * t, z.clone()]
    }
}
