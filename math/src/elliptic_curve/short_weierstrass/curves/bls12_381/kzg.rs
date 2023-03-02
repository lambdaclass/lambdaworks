use crate::{
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
    curve::BLS12381Curve, field_extension::BLS12381FieldConfig, twist::BLS12381TwistCurve,
};

#[derive(Clone, Debug)]
pub struct FrConfig;
impl IsMontgomeryConfiguration for FrConfig {
    // TODO: use U256 when available
    const MODULUS: U384 =
        U384::from("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
}

pub type Fr = MontgomeryBackendPrimeField<BLS12381FieldConfig>;
pub type FrElement = FieldElement<Fr>;

type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

struct Opening {
    value: FrElement,
    proof: G1,
}

struct StructuredReferenceString<const MAXIMUM_DEGREE: usize> {
    powers_main_subgroup: [G1; MAXIMUM_DEGREE],
    generator_secondary_group: G2,
}

struct KZG<const MAXIMUM_DEGREE: usize> {
    srs: StructuredReferenceString<MAXIMUM_DEGREE>,
}

impl<const MAXIMUM_DEGREE: usize> KZG<MAXIMUM_DEGREE> {
    fn commit(p: &Polynomial<Fr>) -> G1 {
        todo!()
    }

    fn open(x: &FrElement, p: &Polynomial<Fr>) -> Opening {
        todo!()
    }

    fn verify(opening: &Opening, x: &FrElement, commitment: &G1) -> Opening {
        todo!()
    }
}
