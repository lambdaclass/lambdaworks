use lambdaworks_crypto::commitments::kzg::{KateZaveruchaGoldberg, StructuredReferenceString};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing, traits::IsPairing,
    },
};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::montgomery_backed_prime_fields::{IsMontgomeryConfiguration, U256PrimeField},
    },
    polynomial::Polynomial,
    unsigned_integer::element::U256,
};

use crate::config::{
    FrConfig, FrElement, FrField, G1Point, G2Point, KZG, NUMBER_CONSTRAINTS, ORDER_4_ROOT_UNITY,
    ORDER_R_MINUS_1_ROOT_UNITY, SRS,
};

type VariableID = u64;

// TODO: implement getters
pub struct Circuit {
    pub number_public_inputs: u64,
    pub number_private_inputs: u64,
    pub number_internal_variables: u64,
}

// TODO: implement getters
pub struct Witness {
    pub a: Vec<FrElement>,
    pub b: Vec<FrElement>,
    pub c: Vec<FrElement>,
}

impl Circuit {
    // TODO: This should execute the program and return the trace.
    pub fn get_witness(&self) -> Witness {
        Witness {
            a: vec![
                FrElement::from(2_u64),
                FrElement::from(4_u64),
                FrElement::from(2_u64),
                FrElement::from(4_u64),
            ],
            b: vec![
                FrElement::zero(),
                FrElement::one(),
                FrElement::from(2_u64),
                FrElement::from(4_u64),
            ],
            c: vec![
                FrElement::zero(),
                FrElement::one(),
                FrElement::from(4_u64),
                FrElement::from(0_u64),
            ],
        }
    }
}

// TODO: implement getters
pub struct CommonPreprocessedInput {
    pub number_constraints: usize,
    pub domain: Vec<FrElement>,

    pub Ql: Polynomial<FrElement>,
    pub Qr: Polynomial<FrElement>,
    pub Qo: Polynomial<FrElement>,
    pub Qm: Polynomial<FrElement>,
    pub Qc: Polynomial<FrElement>,

    pub S1_monomial: Polynomial<FrElement>,
    pub S2_monomial: Polynomial<FrElement>,
    pub S3_monomial: Polynomial<FrElement>,

    pub S1_lagrange: Vec<FrElement>,
    pub S2_lagrange: Vec<FrElement>,
    pub S3_lagrange: Vec<FrElement>,
}

impl CommonPreprocessedInput {
    pub fn for_this(c: &Circuit) -> CommonPreprocessedInput {
        let w = ORDER_4_ROOT_UNITY;
        let u = ORDER_R_MINUS_1_ROOT_UNITY;
        let domain = (1..4).fold(vec![FrElement::one()], |mut acc, x| {
            acc.push(acc.last().unwrap() * ORDER_4_ROOT_UNITY);
            acc
        });

        let S1_lagrange = vec![
            u.pow(2_u64) * w.pow(3_u64),
            u.pow(0_u64) * w.pow(3_u64),
            u.pow(0_u64) * w.pow(0_u64),
            u.pow(0_u64) * w.pow(1_u64),
        ];
        let S2_lagrange = vec![
            u.pow(0_u64) * w.pow(2_u64),
            u.pow(1_u64) * w.pow(0_u64),
            u.pow(1_u64) * w.pow(2_u64),
            u.pow(2_u64) * w.pow(2_u64),
        ];
        let S3_lagrange = vec![
            u.pow(1_u64) * w.pow(1_u64),
            u.pow(2_u64) * w.pow(0_u64),
            u.pow(1_u64) * w.pow(3_u64),
            u.pow(2_u64) * w.pow(1_u64),
        ];

        Self {
            number_constraints: NUMBER_CONSTRAINTS,
            domain: domain.clone(),

            Ql: Polynomial::interpolate(
                &domain,
                &[
                    -FrElement::one(),
                    -FrElement::one(),
                    FrElement::zero(),
                    FrElement::one(),
                ],
            ),
            Qr: Polynomial::interpolate(
                &domain,
                &[
                    FrElement::zero(),
                    FrElement::zero(),
                    FrElement::zero(),
                    -FrElement::one(),
                ],
            ),
            Qo: Polynomial::interpolate(
                &domain,
                &[
                    FrElement::zero(),
                    FrElement::zero(),
                    -FrElement::one(),
                    FrElement::zero(),
                ],
            ),
            Qm: Polynomial::interpolate(
                &domain,
                &[
                    FrElement::zero(),
                    FrElement::zero(),
                    FrElement::one(),
                    FrElement::zero(),
                ],
            ),
            Qc: Polynomial::interpolate(
                &domain,
                &[
                    FrElement::zero(),
                    FrElement::zero(),
                    FrElement::zero(),
                    FrElement::zero(),
                ],
            ),

            S1_monomial: Polynomial::interpolate(&domain, &S1_lagrange),
            S2_monomial: Polynomial::interpolate(&domain, &S2_lagrange),
            S3_monomial: Polynomial::interpolate(&domain, &S3_lagrange),

            S1_lagrange: S1_lagrange,
            S2_lagrange: S2_lagrange,
            S3_lagrange: S3_lagrange,
        }
    }
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

fn build_permutation(circuit: &Circuit) -> Vec<VariableID> {
    todo!()
}

#[allow(unused)]
fn setup<const MAXIMUM_DEGREE: usize, P: IsPairing>(
    srs: &SRS,
    circuit: &Circuit,
) -> VerificationKey<G1Point> {
    let kzg = KZG::new(srs.clone());
    let common_input = CommonPreprocessedInput::for_this(&circuit);

    VerificationKey {
        Qm: kzg.commit(&common_input.Qm),
        Ql: kzg.commit(&common_input.Ql),
        Qr: kzg.commit(&common_input.Qr),
        Qo: kzg.commit(&common_input.Qo),
        Qc: kzg.commit(&common_input.Qc),

        S1: kzg.commit(&common_input.S1_monomial),
        S2: kzg.commit(&common_input.S2_monomial),
        S3: kzg.commit(&common_input.S3_monomial),
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::{
        short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, twist::BLS12381TwistCurve},
        traits::IsEllipticCurve,
    };

    use super::*;
    use crate::{
        config::{FpElement, Pairing},
        test_utils::{test_circuit, test_srs},
    };

    #[test]
    fn setup_works_for_simple_circuit() {
        let srs = test_srs();
        let circuit = test_circuit();

        let vk = setup::<4, BLS12381AtePairing>(&srs, &circuit);

        let expected_Ql = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("1492341357755e31a6306abf3237f84f707ded7cb526b8ffd40901746234ef27f12bc91ef638e4977563db208b765f12"),
            FpElement::from_hex("ec3ff8288ea339010658334f494a614f7470c19a08d53a9cf5718e0613bb65d2cdbc1df374057d9b45c35cf1f1b5b72"),
        ).unwrap();
        let expected_Qr = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("107ab09b6b8c6fc55087aeb8045e17a6d016bdacbc64476264328e71f3e85a4eacaee34ee963e9c9249b6b1bc9653674"),
            FpElement::from_hex("f98e3fe5a53545b67a51da7e7a6cedc51af467abdefd644113fb97edf339aeaa5e2f6a5713725ec76754510b76a10be"),
        ).unwrap();
        let expected_Qo = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("9fd00baa112a0064ce5c3c2d243e657b25df8a2f237b91eec27e83157f6ca896a2401d07ec7d7d097d2f2a344e2018f"),
            FpElement::from_hex("15922cfa65972d80823c6bb9aeb0637c864b636267bfee2818413e9cdc5f7948575c4ce097bb8b9db8087c4ed5056592"),
        ).unwrap();
        let expected_Qm = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("9fd00baa112a0064ce5c3c2d243e657b25df8a2f237b91eec27e83157f6ca896a2401d07ec7d7d097d2f2a344e2018f"),
            FpElement::from_hex("46ee4efd3e8b919c8df3bfc949b495ade2be8228bc524974eef94041a517cdbc74fb31e1998746201f683b12afa4519"),
        ).unwrap();

        let expected_S1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("187ee12de08728650d18912aa9fe54863922a9eeb37e34ff43041f1d039f00028ad2cdf23705e6f6ab7ea9406535c1b0"),
            FpElement::from_hex("4f29051990de0d12b38493992845d9abcb48ef18239eca8b8228618c78ec371d39917bc0d45cf6dc4f79bd64baa9ae2")
        ).unwrap();
        let expected_S2 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("167c0384025887c01ea704234e813842a4acef7d765c3a94a5442ca685b4fc1d1b425ba7786a7413bd4a7d6a1eb5a35a"),
            FpElement::from_hex("12b644100c5d00af27c121806c4779f88e840ff3fdac44124b8175a303d586c4d910486f909b37dda1505c485f053da1")
        ).unwrap();
        let expected_S3 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("188fb6dba3cf5af8a7f6a44d935bb3dd2083a5beb4c020f581739ebc40659c824a4ca8279cf7d852decfbca572e4fa0e"),
            FpElement::from_hex("d84d52582fd95bfa7672f7cef9dd4d0b1b4a54d33f244fdb97df71c7d45fd5c5329296b633c9ed23b8475ee47b9d99")
        ).unwrap();

        assert_eq!(vk.Ql, expected_Ql);
        assert_eq!(vk.Qr, expected_Qr);
        assert_eq!(vk.Qo, expected_Qo);
        assert_eq!(vk.Qm, expected_Qm);

        assert_eq!(vk.S1, expected_S1);
        assert_eq!(vk.S2, expected_S2);
        assert_eq!(vk.S3, expected_S3);
    }
}
