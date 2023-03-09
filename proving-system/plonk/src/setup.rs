use lambdaworks_math::{elliptic_curve::{short_weierstrass::curves::bls12_381::{pairing::BLS12381AtePairing}, traits::IsPairing}, cyclic_group::IsGroup};
use lambdaworks_crypto::commitments::kzg::{KateZaveruchaGoldberg, StructuredReferenceString};
use lambdaworks_math::{polynomial::Polynomial, field::{element::FieldElement, fields::montgomery_backed_prime_fields::{U256PrimeField, IsMontgomeryConfiguration}}, unsigned_integer::element::U256};

use crate::config::{FrElement, G1Point, G2Point, KZG, FrField, FrConfig, ROOT_UNITY, SRS};

pub struct Circuit {
    number_public_inputs: u64,
    number_private_inputs: u64
}

impl Circuit {
    // TODO: This should execute the program and return the trace.
    pub fn get_program_trace_polynomials(&self) -> (Polynomial<FrElement>, Polynomial<FrElement>, Polynomial<FrElement>) {
        (
            Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Polynomial::new(&[FrElement::zero(), FrElement::one()])
        )
    }

    pub fn get_common_preprocessed_input(&self) -> CommonPreprocessedInput {
        CommonPreprocessedInput {
            Ql: vec![-FrElement::one(), -FrElement::one(), FrElement::zero(), FrElement::one()],
            Qr: vec![FrElement::zero(), FrElement::zero(), FrElement::zero(), -FrElement::one()],
            Qo: vec![FrElement::zero(), FrElement::zero(), -FrElement::one(), FrElement::zero()],
            Qm: vec![FrElement::zero(), FrElement::zero(), FrElement::one(), FrElement::zero()],
            Qc: vec![FrElement::zero(), FrElement::zero(), FrElement::zero(), FrElement::zero()],
            S1: vec![FrElement::zero(), FrElement::zero(), FrElement::zero(), FrElement::zero()],
            S2: vec![FrElement::zero(), FrElement::zero(), FrElement::zero(), FrElement::zero()],
            S3: vec![FrElement::zero(), FrElement::zero(), FrElement::zero(), FrElement::zero()],
        }
    }
}

pub struct CommonPreprocessedInput {
    Qm: Vec<FrElement>,
    Ql: Vec<FrElement>,
    Qr: Vec<FrElement>,
    Qo: Vec<FrElement>,
    Qc: Vec<FrElement>,
    S1: Vec<FrElement>,
    S2: Vec<FrElement>,
    S3: Vec<FrElement>
}

pub struct VerificationKey<G1Point> {
    Qm: G1Point,
    Ql: G1Point,
    Qr: G1Point,
    Qo: G1Point,
    Qc: G1Point,
    S1: G1Point,
    S2: G1Point,
    S3: G1Point,
}


#[allow(unused)]
fn setup<const MAXIMUM_DEGREE: usize, P: IsPairing> (
    srs: &SRS,
    circuit: &Circuit
) -> VerificationKey<G1Point> {
    let kzg = KZG::new(srs.clone());
    let common_input = circuit.get_common_preprocessed_input();
    let base: Vec<FrElement> = (0..7).map(|exp| ROOT_UNITY.pow(exp as u64)).collect();

    VerificationKey {
        Qm: kzg.commit(&Polynomial::interpolate(&base, &common_input.Qm)),
        Ql: kzg.commit(&Polynomial::interpolate(&base, &common_input.Ql)),
        Qr: kzg.commit(&Polynomial::interpolate(&base, &common_input.Qr)),
        Qo: kzg.commit(&Polynomial::interpolate(&base, &common_input.Qo)),
        Qc: kzg.commit(&Polynomial::interpolate(&base, &common_input.Qc)),
        S1: kzg.commit(&Polynomial::interpolate(&base, &common_input.S1)),
        S2: kzg.commit(&Polynomial::interpolate(&base, &common_input.S2)),
        S3: kzg.commit(&Polynomial::interpolate(&base, &common_input.S3)),
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::{short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, twist::BLS12381TwistCurve}, traits::IsEllipticCurve};

    use crate::config::{Pairing, FpElement};
    use super::*;

    fn test_srs() -> SRS {
        let s = FrElement::new(U256 { limbs: [1, 0, 0, 0] });
        let g1: G1Point = <BLS12381Curve as IsEllipticCurve>::generator();
        let g2: G2Point = <BLS12381TwistCurve as IsEllipticCurve>::generator();
        
        let powers_main_group: Vec<G1Point> = (0..8).map(|exp| g1.operate_with_self(s.pow(exp as u64).representative())).collect(); 
        let powers_secondary_group = [g2.clone(), g2.operate_with_self(s.representative())];

        SRS::new(&powers_main_group, &powers_secondary_group)
    }

    #[test]
    fn setup_works_for_simple_circuit() {
        let srs = test_srs();
        let circuit = Circuit { number_public_inputs: 2, number_private_inputs: 1 };

        let vk = setup::<100, BLS12381AtePairing>(&srs, &circuit);

        let expected_Qm = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("e04f7ca817bad0b7e83a3225f1ceda46f5caba64fac8eeab76745db6ff470da1da58e0e887410b47540a13cff5f9672"),
            FpElement::from_hex("179d2cad1f9ebe842143f93681beee061641c4d054f129ff49eb637c5cf638cb7dde1177dc58065747affe7587d25f7"),
        ).unwrap();
                
        assert_eq!(vk.Qm, expected_Qm);
    }
}