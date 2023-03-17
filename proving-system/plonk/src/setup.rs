
use lambdaworks_crypto::commitments::{kzg::{StructuredReferenceString, KateZaveruchaGoldberg}, traits::IsCommitmentScheme};
use lambdaworks_math::{
    elliptic_curve::{
        traits::IsPairing,
    }, field::{traits::IsField, element::FieldElement},
};
use lambdaworks_math::{
    polynomial::Polynomial,
};

use crate::test_utils::{ORDER_4_ROOT_UNITY, ORDER_R_MINUS_1_ROOT_UNITY, NUMBER_CONSTRAINTS};


type VariableID = u64;

// TODO: implement getters
pub struct Circuit {
    pub number_public_inputs: u64,
    pub number_private_inputs: u64,
    pub number_internal_variables: u64,
}

// TODO: implement getters
pub struct Witness<F: IsField> {
    pub a: Vec<FieldElement<F>>,
    pub b: Vec<FieldElement<F>>,
    pub c: Vec<FieldElement<F>>,
}

impl Circuit {
    // TODO: This should execute the program and return the trace.
    pub fn get_witness<F: IsField>(&self) -> Witness<F> {
        Witness {
            a: vec![
                FieldElement::from(2_u64),
                FieldElement::from(4_u64),
                FieldElement::from(2_u64),
                FieldElement::from(4_u64),
            ],
            b: vec![
                FieldElement::from(2_u64),
                FieldElement::from(2_u64),
                FieldElement::from(2_u64),
                FieldElement::from(4_u64),
            ],
            c: vec![
                FieldElement::from(2_u64),
                FieldElement::from(2_u64),
                FieldElement::from(4_u64),
                FieldElement::from(2_u64),
            ],
        }
    }
}

// TODO: implement getters
pub struct CommonPreprocessedInput<F: IsField> {
    pub number_constraints: usize,
    pub domain: Vec<FieldElement<F>>,
    pub order_4_root_unity: FieldElement<F>,
    pub order_r_minus_1_root_unity: FieldElement<F>,

    pub Ql: Polynomial<FieldElement<F>>,
    pub Qr: Polynomial<FieldElement<F>>,
    pub Qo: Polynomial<FieldElement<F>>,
    pub Qm: Polynomial<FieldElement<F>>,
    pub Qc: Polynomial<FieldElement<F>>,

    pub S1_monomial: Polynomial<FieldElement<F>>,
    pub S2_monomial: Polynomial<FieldElement<F>>,
    pub S3_monomial: Polynomial<FieldElement<F>>,

    pub S1_lagrange: Vec<FieldElement<F>>,
    pub S2_lagrange: Vec<FieldElement<F>>,
    pub S3_lagrange: Vec<FieldElement<F>>,
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
fn setup<F: IsField, CS: IsCommitmentScheme<F>>(
    common_input: &CommonPreprocessedInput<F>,
    commitment_scheme: &CS,
    circuit: &Circuit,
) -> VerificationKey<CS::Hiding> {
    VerificationKey {
        Qm: commitment_scheme.commit(&common_input.Qm),
        Ql: commitment_scheme.commit(&common_input.Ql),
        Qr: commitment_scheme.commit(&common_input.Qr),
        Qo: commitment_scheme.commit(&common_input.Qo),
        Qc: commitment_scheme.commit(&common_input.Qc),

        S1: commitment_scheme.commit(&common_input.S1_monomial),
        S2: commitment_scheme.commit(&common_input.S2_monomial),
        S3: commitment_scheme.commit(&common_input.S3_monomial),
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::{
        short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, pairing::BLS12381AtePairing},
        traits::IsEllipticCurve,
    };

    use super::*;
    use crate::{
        test_utils::{FpElement, test_circuit, test_srs, test_common_preprocessed_input, FrField, KZG},
    };

    #[test]
    fn setup_works_for_simple_circuit() {
        let srs = test_srs();
        let circuit = test_circuit();
        let common_input = test_common_preprocessed_input();
        let kzg = KZG::new(srs);

        let vk = setup::<FrField, KZG>(&common_input, &kzg, &circuit);

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
