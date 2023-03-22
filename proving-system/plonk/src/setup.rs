use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
};

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
    pub n: usize,
    /// Number of constraints
    pub domain: Vec<FieldElement<F>>,
    pub omega: FieldElement<F>, // Order 4 root unity
    pub k1: FieldElement<F>,    // Order R minus one root unity

    pub ql: Polynomial<FieldElement<F>>,
    pub qr: Polynomial<FieldElement<F>>,
    pub qo: Polynomial<FieldElement<F>>,
    pub qm: Polynomial<FieldElement<F>>,
    pub qc: Polynomial<FieldElement<F>>,

    pub s1: Polynomial<FieldElement<F>>,
    pub s2: Polynomial<FieldElement<F>>,
    pub s3: Polynomial<FieldElement<F>>,

    pub s1_lagrange: Vec<FieldElement<F>>,
    pub s2_lagrange: Vec<FieldElement<F>>,
    pub s3_lagrange: Vec<FieldElement<F>>,
}

#[allow(unused)]
pub struct VerificationKey<G1Point> {
    pub qm_1: G1Point,
    pub ql_1: G1Point,
    pub qr_1: G1Point,
    pub qo_1: G1Point,
    pub qc_1: G1Point,

    pub s1_1: G1Point,
    pub s2_1: G1Point,
    pub s3_1: G1Point,
}

#[allow(unused)]
pub fn setup<F: IsField, CS: IsCommitmentScheme<F>>(
    common_input: &CommonPreprocessedInput<F>,
    commitment_scheme: &CS,
    circuit: &Circuit,
) -> VerificationKey<CS::Commitment> {
    VerificationKey {
        qm_1: commitment_scheme.commit(&common_input.qm),
        ql_1: commitment_scheme.commit(&common_input.ql),
        qr_1: commitment_scheme.commit(&common_input.qr),
        qo_1: commitment_scheme.commit(&common_input.qo),
        qc_1: commitment_scheme.commit(&common_input.qc),

        s1_1: commitment_scheme.commit(&common_input.s1),
        s2_1: commitment_scheme.commit(&common_input.s2),
        s3_1: commitment_scheme.commit(&common_input.s3),
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::{
        short_weierstrass::curves::bls12_381::curve::BLS12381Curve, traits::IsEllipticCurve,
    };

    use super::*;
    use crate::test_utils::{
        test_circuit, test_common_preprocessed_input, test_srs, FpElement, FrField, KZG,
    };

    #[test]
    fn setup_works_for_simple_circuit() {
        let srs = test_srs();
        let circuit = test_circuit();
        let common_input = test_common_preprocessed_input();
        let kzg = KZG::new(srs);

        let vk = setup::<FrField, KZG>(&common_input, &kzg, &circuit);

        let expected_ql = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("1492341357755e31a6306abf3237f84f707ded7cb526b8ffd40901746234ef27f12bc91ef638e4977563db208b765f12"),
            FpElement::from_hex("ec3ff8288ea339010658334f494a614f7470c19a08d53a9cf5718e0613bb65d2cdbc1df374057d9b45c35cf1f1b5b72"),
        ).unwrap();
        let expected_qr = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("107ab09b6b8c6fc55087aeb8045e17a6d016bdacbc64476264328e71f3e85a4eacaee34ee963e9c9249b6b1bc9653674"),
            FpElement::from_hex("f98e3fe5a53545b67a51da7e7a6cedc51af467abdefd644113fb97edf339aeaa5e2f6a5713725ec76754510b76a10be"),
        ).unwrap();
        let expected_qo = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("9fd00baa112a0064ce5c3c2d243e657b25df8a2f237b91eec27e83157f6ca896a2401d07ec7d7d097d2f2a344e2018f"),
            FpElement::from_hex("15922cfa65972d80823c6bb9aeb0637c864b636267bfee2818413e9cdc5f7948575c4ce097bb8b9db8087c4ed5056592"),
        ).unwrap();
        let expected_qm = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("9fd00baa112a0064ce5c3c2d243e657b25df8a2f237b91eec27e83157f6ca896a2401d07ec7d7d097d2f2a344e2018f"),
            FpElement::from_hex("46ee4efd3e8b919c8df3bfc949b495ade2be8228bc524974eef94041a517cdbc74fb31e1998746201f683b12afa4519"),
        ).unwrap();

        let expected_s1 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("187ee12de08728650d18912aa9fe54863922a9eeb37e34ff43041f1d039f00028ad2cdf23705e6f6ab7ea9406535c1b0"),
            FpElement::from_hex("4f29051990de0d12b38493992845d9abcb48ef18239eca8b8228618c78ec371d39917bc0d45cf6dc4f79bd64baa9ae2")
        ).unwrap();
        let expected_s2 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("167c0384025887c01ea704234e813842a4acef7d765c3a94a5442ca685b4fc1d1b425ba7786a7413bd4a7d6a1eb5a35a"),
            FpElement::from_hex("12b644100c5d00af27c121806c4779f88e840ff3fdac44124b8175a303d586c4d910486f909b37dda1505c485f053da1")
        ).unwrap();
        let expected_s3 = BLS12381Curve::create_point_from_affine(
            FpElement::from_hex("188fb6dba3cf5af8a7f6a44d935bb3dd2083a5beb4c020f581739ebc40659c824a4ca8279cf7d852decfbca572e4fa0e"),
            FpElement::from_hex("d84d52582fd95bfa7672f7cef9dd4d0b1b4a54d33f244fdb97df71c7d45fd5c5329296b633c9ed23b8475ee47b9d99")
        ).unwrap();

        assert_eq!(vk.ql_1, expected_ql);
        assert_eq!(vk.qr_1, expected_qr);
        assert_eq!(vk.qo_1, expected_qo);
        assert_eq!(vk.qm_1, expected_qm);

        assert_eq!(vk.s1_1, expected_s1);
        assert_eq!(vk.s2_1, expected_s2);
        assert_eq!(vk.s3_1, expected_s3);
    }
}
