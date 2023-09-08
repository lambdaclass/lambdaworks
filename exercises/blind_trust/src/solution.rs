use lambdaworks_crypto::{
    commitments::traits::IsCommitmentScheme, fiat_shamir::transcript::Transcript,
};
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    traits::{ByteConversion, Serializable},
};
use lambdaworks_plonk::{
    prover::Proof,
    setup::{new_strong_fiat_shamir_transcript, CommonPreprocessedInput, VerificationKey},
};

fn compute_private_input<F, CS>(
    proof: &Proof<F, CS>,
    vk: &VerificationKey<CS::Commitment>,
    public_input: &[FieldElement<F>],
    common_preprocessed_input: &CommonPreprocessedInput<F>,
) -> (FieldElement<F>, FieldElement<F>)
where
    F: IsField,
    CS: IsCommitmentScheme<F>,
    CS::Commitment: Serializable,
    FieldElement<F>: ByteConversion,
{
    // Replay interactions to recover challenges
    let mut transcript = new_strong_fiat_shamir_transcript::<F, CS>(vk, public_input);
    transcript.append(&proof.a_1.serialize());
    transcript.append(&proof.b_1.serialize());
    transcript.append(&proof.c_1.serialize());
    let _beta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
    let _gamma = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    transcript.append(&proof.z_1.serialize());
    let _alpha = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    transcript.append(&proof.t_lo_1.serialize());
    transcript.append(&proof.t_mid_1.serialize());
    transcript.append(&proof.t_hi_1.serialize());
    let zeta = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();

    // Compute `x` and `h`
    let [b, y] = [&public_input[0], &public_input[1]];
    let n = common_preprocessed_input.n as u64;
    let omega = &common_preprocessed_input.omega;
    let domain = &common_preprocessed_input.domain;
    let l1_zeta =
        (zeta.pow(n) - FieldElement::one()) / (&zeta - FieldElement::one()) / FieldElement::from(n);

    let mut li_zeta = l1_zeta;
    let mut lagrange_basis_zeta = Vec::new();
    lagrange_basis_zeta.push(li_zeta.clone());
    for i in 1..domain.len() {
        li_zeta = omega * &li_zeta * ((&zeta - &domain[i - 1]) / (&zeta - &domain[i]));
        lagrange_basis_zeta.push(li_zeta.clone());
    }

    let x = (&proof.a_zeta
        - b * &lagrange_basis_zeta[3]
        - y * &lagrange_basis_zeta[4]
        - b * &lagrange_basis_zeta[0]
        - y * &lagrange_basis_zeta[1]
        - b * &lagrange_basis_zeta[5]
        - b * &lagrange_basis_zeta[6]
        - b * &lagrange_basis_zeta[7])
        / &lagrange_basis_zeta[2];
    let h = (y - b) / &x;
    (x, h)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{BufReader, Read},
    };

    use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;
    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement,
        traits::{Deserializable, Serializable},
    };
    use lambdaworks_plonk::{prover::Proof, setup::setup, verifier::Verifier};

    use crate::{
        circuit::circuit_common_preprocessed_input,
        sith_generate_proof::{
            generate_proof, SithProof, SithSRS, H_COORDINATE, KZG, X_COORDINATE,
        },
        solution::compute_private_input,
    };

    fn read_challenge_data_from_files() -> (SithSRS, SithProof) {
        // Read proof from file
        let f = fs::File::open("./proof").unwrap();
        let mut reader = BufReader::new(f);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();
        let proof = Proof::deserialize(&buffer).unwrap();

        // Read SRS from file
        let f = fs::File::open("./srs").unwrap();
        let mut reader = BufReader::new(f);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).unwrap();
        let srs = StructuredReferenceString::deserialize(&buffer).unwrap();
        (srs, proof)
    }

    #[test]
    fn test_challenge_data() {
        let b =
            FrElement::from_hex("1b0871ce73e72c599426228e37e7469be9f4fa0b7c9dae950bb77539ca9ebb0f")
                .unwrap();
        let y =
            FrElement::from_hex("3610e39ce7acc430c1fa91efcec93722d77bc4e910ccb195fa4294b64ecb0d35")
                .unwrap();
        let public_input = vec![b, y];

        let (srs, proof) = read_challenge_data_from_files();
        let common_preprocessed_input = circuit_common_preprocessed_input();
        let kzg = KZG::new(srs.clone());
        let verifier = Verifier::new(kzg.clone());
        let vk = setup(&common_preprocessed_input, &kzg);

        assert!(verifier.verify(&proof, &public_input, &common_preprocessed_input, &vk))
    }

    fn export_challenge_data() {
        use std::fs;
        use std::io::Write;

        let b =
            FrElement::from_hex("1b0871ce73e72c599426228e37e7469be9f4fa0b7c9dae950bb77539ca9ebb0f")
                .unwrap();
        let (y, proof, srs) = generate_proof(&b);

        let mut srs_file = fs::File::create("./srs").unwrap();
        srs_file.write_all(&srs.serialize()).unwrap();
        let mut srs_file = fs::File::create("./proof").unwrap();
        srs_file.write_all(&proof.serialize()).unwrap();
        println!("{}", y);
    }

    #[test]
    fn test_solution() {
        let b =
            FrElement::from_hex("1b0871ce73e72c599426228e37e7469be9f4fa0b7c9dae950bb77539ca9ebb0f")
                .unwrap();
        let y =
            FrElement::from_hex("3610e39ce7acc430c1fa91efcec93722d77bc4e910ccb195fa4294b64ecb0d35")
                .unwrap();
        let public_input = vec![b, y];

        let (srs, proof) = read_challenge_data_from_files();
        let common_preprocessed_input = circuit_common_preprocessed_input();
        let kzg = KZG::new(srs.clone());

        let vk = setup(&common_preprocessed_input, &kzg);
        // Extract private input from proof, public input and public keys
        let (x, h) = compute_private_input(&proof, &vk, &public_input, &common_preprocessed_input);

        assert_eq!(&X_COORDINATE, &x);
        assert_eq!(&H_COORDINATE, &h);
    }
}
