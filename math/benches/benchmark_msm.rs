use criterion::{BenchmarkId, Criterion};
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

pub fn msm_with_size(c: &mut Criterion, msm_size: usize) {
    // The following are fixed window_sizes to be tested for the
    // provided msm_size. I currently found them to provide good coverage,
    // but when extending these benchmark to real field/curves we might want to
    // tune this further.
    let window_sizes = vec![1, 2, 3, 4];

    // We boostrap a rng with a fixed seed so the benchmarks are reproducible.
    let mut rng = StdRng::seed_from_u64(42);
    let g = EllipticCurveElement::<CurrentCurve>::generator();

    // Generate field elements and group elements with the available field/curves
    // that we have today, which are quite limited. We should make this a further
    // benchmark dimension testing for different real field and curves.
    let s: Vec<FE> = (0..msm_size).map(|_| FE::new(rng.gen())).collect();
    let hiding: Vec<EllipticCurveElement<CurrentCurve>> = (0..msm_size)
        .map(|x| g.operate_with_self(x as u128))
        .collect();

    let mut group = c.benchmark_group(format!("msm_{}", msm_size));

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
