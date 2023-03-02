use core::num;

use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint,
    field::{
        element::FieldElement,
        fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
    },
    msm::msm,
    polynomial::Polynomial,
    unsigned_integer::element::U384,
};

use super::{
    curve::BLS12381Curve,
    pairing::{ate, batch_ate},
    twist::BLS12381TwistCurve,
};

#[derive(Clone, Debug)]
pub struct FrConfig;
impl IsMontgomeryConfiguration for FrConfig {
    // TODO: use U256 when available
    const MODULUS: U384 =
        U384::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
}

pub type FrElement = FieldElement<MontgomeryBackendPrimeField<FrConfig>>;

type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

struct Opening {
    value: FrElement,
    proof: G1,
}

struct StructuredReferenceString<const MAXIMUM_DEGREE: usize> {
    powers_main_group: Vec<G1>,
    powers_secondary_group: [G2; 2],
}

impl<const MAXIMUM_DEGREE: usize> StructuredReferenceString<MAXIMUM_DEGREE> {
    pub fn new(powers_main_group: &[G1], powers_secondary_group: &[G2; 2]) -> Self {
        Self {
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }
}

struct KZG<const MAXIMUM_DEGREE: usize> {
    srs: StructuredReferenceString<MAXIMUM_DEGREE>,
}

impl<const MAXIMUM_DEGREE: usize> KZG<MAXIMUM_DEGREE> {
    fn new(srs: StructuredReferenceString<MAXIMUM_DEGREE>) -> Self {
        Self { srs }
    }
    fn commit(&self, p: &Polynomial<FrElement>) -> G1 {
        let coefficients: Vec<U384> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        msm(
            &coefficients,
            &self.srs.powers_main_group[..coefficients.len()],
        )
    }

    fn open(&self, x: &FrElement, p: &Polynomial<FrElement>) -> Opening {
        let value = p.evaluate(&x);
        let numerator = p + Polynomial::new_monomial(-&value, 0);
        let denominator = Polynomial::new(&[-x, FieldElement::one()]);
        let proof = self.commit(&(numerator / denominator));

        Opening { value, proof }
    }

    fn verify(&self, opening: &Opening, x: &FrElement, p_commitment: &G1) -> bool {
        let g1 = &self.srs.powers_main_group[0];
        let g2 = &self.srs.powers_secondary_group[0];
        let alpha_g2 = &self.srs.powers_secondary_group[1];

        let e = batch_ate(&[
            (&p_commitment
                .operate_with(&(g1.operate_with_self(opening.value.representative())).neg()).to_affine(),
            &g2.to_affine()),
            (&opening.proof.neg().to_affine(),
            &(alpha_g2.operate_with(&(g2.operate_with_self(x.representative())).neg())).to_affine()),
        ]);
        e == FieldElement::one()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::{
                curve::BLS12381Curve, twist::BLS12381TwistCurve,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        polynomial::Polynomial,
        unsigned_integer::element::U384,
    };

    use super::{FrElement, StructuredReferenceString, G1, KZG};
    use rand::Rng;

    fn create_srs() -> StructuredReferenceString<100> {
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
            (&g2).clone(),
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
