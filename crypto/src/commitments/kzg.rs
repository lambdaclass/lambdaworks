use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::IsPairing,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::msm,
    polynomial::Polynomial,
};
use std::marker::PhantomData;

pub struct Opening<F: IsPrimeField, G1Point: IsGroup> {
    value: FieldElement<F>,
    proof: G1Point,
}

#[derive(Clone)]
pub struct StructuredReferenceString<const MAXIMUM_DEGREE: usize, G1Point, G2Point> {
    pub powers_main_group: Vec<G1Point>,
    pub powers_secondary_group: [G2Point; 2],
}

impl<const MAXIMUM_DEGREE: usize, G1Point, G2Point>
    StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>
where
    G1Point: IsGroup,
    G2Point: IsGroup,
{
    #[allow(unused)]
    pub fn new(powers_main_group: &[G1Point], powers_secondary_group: &[G2Point; 2]) -> Self {
        Self {
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }
}

pub struct KateZaveruchaGoldberg<const MAXIMUM_DEGREE: usize, F: IsPrimeField, P: IsPairing> {
    srs: StructuredReferenceString<MAXIMUM_DEGREE, P::G1Point, P::G2Point>,
    phantom: PhantomData<F>,
}

impl<const MAXIMUM_DEGREE: usize, F: IsPrimeField, P: IsPairing>
    KateZaveruchaGoldberg<MAXIMUM_DEGREE, F, P>
{
    #[allow(unused)]
    pub fn new(srs: StructuredReferenceString<MAXIMUM_DEGREE, P::G1Point, P::G2Point>) -> Self {
        Self {
            srs,
            phantom: PhantomData,
        }
    }

    #[allow(unused)]
    pub fn commit(&self, p: &Polynomial<FieldElement<F>>) -> P::G1Point {
        let coefficients: Vec<F::RepresentativeType> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        msm(
            &coefficients,
            &self.srs.powers_main_group[..coefficients.len()],
        )
    }

    #[allow(unused)]
    pub fn open(
        &self,
        x: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Opening<F, P::G1Point> {
        let value = p.evaluate(x);
        let numerator = p + Polynomial::new_monomial(-&value, 0);
        let denominator = Polynomial::new(&[-x, FieldElement::one()]);
        let proof = self.commit(&(numerator / denominator));

        Opening { value, proof }
    }

    #[allow(unused)]
    pub fn verify(
        &self,
        opening: &Opening<F, P::G1Point>,
        x: &FieldElement<F>,
        p_commitment: &P::G1Point,
    ) -> bool {
        let g1 = &self.srs.powers_main_group[0];
        let g2 = &self.srs.powers_secondary_group[0];
        let alpha_g2 = &self.srs.powers_secondary_group[1];

        let e = P::compute_batch(&[
            (
                &p_commitment
                    .operate_with(&(g1.operate_with_self(opening.value.representative())).neg()),
                g2,
            ),
            (
                &opening.proof.neg(),
                &(alpha_g2.operate_with(&(g2.operate_with_self(x.representative())).neg())),
            ),
        ]);
        e == FieldElement::one()
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::{
                    curve::BLS12381Curve, pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
                },
                point::ShortWeierstrassProjectivePoint,
            },
            traits::{IsEllipticCurve, IsPairing},
        },
        field::{
            element::FieldElement,
            fields::montgomery_backed_prime_fields::{
                IsMontgomeryConfiguration, MontgomeryBackendPrimeField,
            },
        },
        polynomial::Polynomial,
        unsigned_integer::element::U256,
    };

    use crate::commitments::kzg::Opening;

    use super::{KateZaveruchaGoldberg, StructuredReferenceString};
    use rand::Rng;

    #[derive(Clone, Debug)]
    pub struct FrConfig;
    impl IsMontgomeryConfiguration<4> for FrConfig {
        const MODULUS: U256 =
            U256::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
    }

    type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
    type FrField = MontgomeryBackendPrimeField<FrConfig, 4>;
    type FrElement = FieldElement<FrField>;
    type KZG = KateZaveruchaGoldberg<100, FrField, BLS12381AtePairing>;

    fn create_srs() -> StructuredReferenceString<
        100,
        <BLS12381AtePairing as IsPairing>::G1Point,
        <BLS12381AtePairing as IsPairing>::G2Point,
    > {
        let mut rng = rand::thread_rng();
        let toxic_waste = FrElement::new(U256 {
            limbs: [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ],
        });
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let powers_main_group: Vec<G1> = (0..100)
            .map(|exponent| {
                g1.operate_with_self(toxic_waste.pow(exponent as u128).representative())
            })
            .collect();
        let powers_secondary_group = [
            g2.clone(),
            g2.operate_with_self(toxic_waste.representative()),
        ];
        StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
    }

    #[test]
    fn kzg_1() {
        let kzg = KZG::new(create_srs());
        let p = Polynomial::<FrElement>::new(&[FieldElement::one(), FieldElement::one()]);
        let p_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p);
        let x = -FieldElement::one();
        let opening = kzg.open(&x, &p);
        assert_eq!(opening.value, FieldElement::zero());
        assert_eq!(opening.proof, BLS12381Curve::generator());
        assert!(kzg.verify(&opening, &x, &p_commitment));
    }
}
