use std::marker::PhantomData;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{traits::{IsPairing}},
    field::{
        element::FieldElement,
        traits::IsPrimeField,
    },
    msm::msm,
    polynomial::Polynomial,
};

struct Opening<F: IsPrimeField, P: IsPairing> {
    value: FieldElement<F>,
    proof: P::G1,
}

struct StructuredReferenceString<const MAXIMUM_DEGREE: usize, P: IsPairing> {
    powers_main_group: Vec<P::G1>,
    powers_secondary_group: [P::G2; 2],
}

impl<const MAXIMUM_DEGREE: usize, P: IsPairing> StructuredReferenceString<MAXIMUM_DEGREE, P> {
    pub fn new(powers_main_group: &[P::G1], powers_secondary_group: &[P::G2; 2]) -> Self {
        Self {
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }
}

struct KateZaveruchaGoldberg<const MAXIMUM_DEGREE: usize, F: IsPrimeField, P: IsPairing> {
    srs: StructuredReferenceString<MAXIMUM_DEGREE, P>,
    phantom: PhantomData<F>
}

impl<const MAXIMUM_DEGREE: usize, F: IsPrimeField, P: IsPairing> KateZaveruchaGoldberg<MAXIMUM_DEGREE, F, P> {
    fn new(srs: StructuredReferenceString<MAXIMUM_DEGREE, P>) -> Self {
        Self { srs , phantom: PhantomData }
    }

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> P::G1 {
        let coefficients: Vec<F::BaseUnsignedType> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        msm(
            &coefficients,
            &self.srs.powers_main_group[..coefficients.len()],
        )
    }

    fn open(&self, x: &FieldElement<F>, p: &Polynomial<FieldElement<F>>) -> Opening<F, P> {
        let value = p.evaluate(x);
        let numerator = p + Polynomial::new_monomial(-&value, 0);
        let denominator = Polynomial::new(&[-x, FieldElement::one()]);
        let proof = self.commit(&(numerator / denominator));

        Opening { value, proof }
    }

    fn verify(&self, opening: &Opening<F, P>, x: &FieldElement<F>, p_commitment: &P::G1) -> bool {
        let g1 = &self.srs.powers_main_group[0];
        let g2 = &self.srs.powers_secondary_group[0];
        let alpha_g2 = &self.srs.powers_secondary_group[1];

        let e = P::compute_batch(&[
            (&p_commitment
                .operate_with(&(g1.operate_with_self(opening.value.representative())).neg()),
            g2),
            (&opening.proof.neg(),
            &(alpha_g2.operate_with(&(g2.operate_with_self(x.representative())).neg()))),
        ]);
        e == FieldElement::one()
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{curves::bls12_381::{
                curve::BLS12381Curve, twist::BLS12381TwistCurve, pairing::BLS12381AtePairing,
            }, point::ShortWeierstrassProjectivePoint},
            traits::IsEllipticCurve,
        },
        field::{element::FieldElement, fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField}},
        polynomial::Polynomial,
        unsigned_integer::element::U384,
    };

    use super::{StructuredReferenceString, KateZaveruchaGoldberg as KZG};
    use rand::Rng;

    #[derive(Clone, Debug)]
    pub struct FrConfig;
    impl IsMontgomeryConfiguration for FrConfig {
        // TODO: use U256 when available
        const MODULUS: U384 =
            U384::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
    }
    
    type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
    pub type FrElement = FieldElement<MontgomeryBackendPrimeField<FrConfig>>;

    fn create_srs() -> StructuredReferenceString<100, BLS12381AtePairing> {
        let mut rng = rand::thread_rng();
        let toxic_waste = FrElement::new(U384 {
            limbs: [
                0_u64,
                0_u64,
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
        let p_commitment = kzg.commit(&p);
        let x = -FieldElement::one();
        let opening = kzg.open(&x, &p);
        assert_eq!(opening.value, FieldElement::zero());
        assert_eq!(opening.proof, BLS12381Curve::generator());
        assert!(kzg.verify(&opening, &x, &p_commitment));
    }
}
