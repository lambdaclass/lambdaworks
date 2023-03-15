use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, twist::BLS12381TwistCurve},
        traits::IsEllipticCurve,
    },
};

use crate::{
    config::{FrElement, G1Point, G2Point, SRS},
    setup::Circuit,
};

pub fn test_srs() -> SRS {
    let s = FrElement::from(2);
    let g1: G1Point = <BLS12381Curve as IsEllipticCurve>::generator();
    let g2: G2Point = <BLS12381TwistCurve as IsEllipticCurve>::generator();

    let powers_main_group: Vec<G1Point> = (0..24)
        .map(|exp| g1.operate_with_self(s.pow(exp as u64).representative()))
        .collect();
    let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

    SRS::new(&powers_main_group, &powers_secondary_group)
}

pub fn test_circuit() -> Circuit {
    Circuit {
        number_public_inputs: 2,
        number_private_inputs: 1,
        number_internal_variables: 1,
    }
}
