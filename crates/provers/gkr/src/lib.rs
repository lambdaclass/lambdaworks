pub mod circuit;
pub mod prover;
pub mod sumcheck;
pub mod verifier;
use crate::circuit::Circuit;
use crate::prover::{GKRProof, Prover, ProverError};
use crate::verifier::Verifier;
use crate::verifier::VerifierError;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::traits::ByteConversion;

/// Serialize the circuit, so that it can be appended to the transcript.
/// This function is used at the beginning of the protocol by the prover and the verifier.
pub fn circuit_to_bytes(circuit: &Circuit) -> Vec<u8> {
    let mut bytes = Vec::new();
    // Append the number of layers and the number of inputs.
    bytes.extend_from_slice(&(circuit.layers().len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(circuit.num_inputs() as u32).to_le_bytes());
    // For each layer append the number of gates, the type and the input indeces of each gate.
    for layer in circuit.layers() {
        bytes.extend_from_slice(&(layer.len() as u32).to_le_bytes());
        for gate in &layer.layer {
            let gate_type = match gate.gate_type {
                crate::circuit::GateType::Add => 0u8,
                crate::circuit::GateType::Mul => 1u8,
            };
            bytes.push(gate_type);
            bytes.extend_from_slice(&(gate.inputs[0] as u32).to_le_bytes());
            bytes.extend_from_slice(&(gate.inputs[1] as u32).to_le_bytes());
        }
    }
    bytes
}

/// Evaluate the line polynomial that goes from `b` to `c`, at point `t`:
/// `l(t) = b + t * (c - b)`.
/// `l` satisfies: l(0) = b and l(1) = c.
/// This function is used in the protocol by the prover and the verifier in each layer.
pub fn line<F>(
    b: &[FieldElement<F>],
    c: &[FieldElement<F>],
    t: &FieldElement<F>,
) -> Vec<FieldElement<F>>
where
    F: IsField,
{
    b.iter()
        .zip(c.iter())
        .map(|(b_val, c_val)| b_val.clone() + t.clone() * (c_val.clone() - b_val.clone()))
        .collect()
}

pub fn gkr_prove<F>(
    circuit: &Circuit,
    input: &[FieldElement<F>],
) -> Result<GKRProof<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    Prover::generate_proof(circuit, input)
}

pub fn gkr_verify<F>(proof: &GKRProof<F>, circuit: &Circuit) -> Result<bool, VerifierError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    Verifier::verify(proof, circuit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, CircuitError, CircuitLayer, Gate, GateType};
    use lambdaworks_math::{field::fields::u64_prime_field::U64PrimeField, polynomial::Polynomial};

    const MODULUS: u64 = 389;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    const MODULUS23: u64 = 23;
    type F23 = U64PrimeField<MODULUS23>;
    type F23E = FieldElement<F23>;

    /// Create the circuit from Thaler's book (Figure 4.12)
    fn thaler_book_circuit() -> Result<Circuit, CircuitError> {
        Circuit::new(
            vec![
                CircuitLayer::new(vec![
                    Gate::new(GateType::Mul, [0, 1]),
                    Gate::new(GateType::Mul, [2, 3]),
                ]),
                CircuitLayer::new(vec![
                    Gate::new(GateType::Mul, [0, 0]),
                    Gate::new(GateType::Mul, [1, 1]),
                    Gate::new(GateType::Mul, [1, 2]),
                    Gate::new(GateType::Mul, [3, 3]),
                ]),
            ],
            4,
        )
    }

    /// Create the circuit from our blog post on the GKR protocol.
    /// https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/
    pub fn lambda_post_circuit() -> Result<Circuit, CircuitError> {
        use crate::circuit::{Circuit, CircuitLayer, Gate, GateType};
        Circuit::new(
            vec![
                CircuitLayer::new(vec![
                    Gate::new(GateType::Mul, [0, 1]),
                    Gate::new(GateType::Add, [2, 3]),
                ]),
                CircuitLayer::new(vec![
                    Gate::new(GateType::Mul, [0, 1]),
                    Gate::new(GateType::Add, [0, 0]),
                    Gate::new(GateType::Add, [0, 1]),
                    Gate::new(GateType::Mul, [0, 1]),
                ]),
            ],
            2,
        )
    }
    /// Create a circuit with four layers (without counting the inputs).
    /// To picture this circuit, imagine a tree structure where each layer has twice the number of gates as the layer above.
    pub fn four_layer_circuit() -> Result<Circuit, CircuitError> {
        use crate::circuit::{CircuitLayer, Gate, GateType};
        use GateType::{Add, Mul};

        let l0 = CircuitLayer::new(vec![Gate::new(Mul, [0, 1])]);

        let l1 = CircuitLayer::new(vec![Gate::new(Add, [0, 1]), Gate::new(Mul, [2, 3])]);

        let l2 = CircuitLayer::new(vec![
            Gate::new(Mul, [0, 1]),
            Gate::new(Add, [2, 3]),
            Gate::new(Mul, [4, 5]),
            Gate::new(Add, [6, 7]),
        ]);

        let l3 = CircuitLayer::new(vec![
            Gate::new(Add, [0, 1]),
            Gate::new(Mul, [2, 3]),
            Gate::new(Add, [4, 5]),
            Gate::new(Mul, [6, 7]),
            Gate::new(Add, [8, 9]),
            Gate::new(Mul, [10, 11]),
            Gate::new(Add, [12, 13]),
            Gate::new(Mul, [14, 15]),
        ]);

        Circuit::new(vec![l0, l1, l2, l3], 16)
    }

    #[test]
    fn test_lambda_circuit_evaluation() {
        let circuit = lambda_post_circuit().unwrap();
        let input = [F23E::from(3), F23E::from(1)];
        let evaluation = circuit.evaluate(&input);
        assert_eq!(evaluation.layers.len(), 3);
        assert_eq!(evaluation.layers[0], [F23E::from(18), F23E::from(7)]);
        assert_eq!(
            evaluation.layers[1],
            [F23E::from(3), F23E::from(6), F23E::from(4), F23E::from(3)]
        );
        assert_eq!(evaluation.layers[2], input.to_vec());
    }

    #[test]
    fn test_thaler_book_circuit_evaluation() {
        let circuit = thaler_book_circuit().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        let evaluation = circuit.evaluate(&input);

        assert_eq!(evaluation.layers.len(), 3);
        assert_eq!(evaluation.layers[0], [FE::from(36), FE::from(6)]);
        assert_eq!(
            evaluation.layers[1],
            [FE::from(9), FE::from(4), FE::from(6), FE::from(1)]
        );
        assert_eq!(evaluation.layers[2], input.to_vec());
    }

    #[test]
    fn test_four_layer_circuit_evaluation() {
        let circuit = four_layer_circuit().unwrap();
        let input: Vec<F23E> = (0u64..16u64).map(F23E::from).collect();

        let evaluation = circuit.evaluate(&input);

        assert_eq!(evaluation.layers.len(), 5);

        assert_eq!(evaluation.layers[0], vec![F23E::from(17)]);
        assert_eq!(evaluation.layers[1], vec![F23E::from(11), F23E::from(12)]);
        assert_eq!(
            evaluation.layers[2],
            vec![F23E::from(6), F23E::from(5), F23E::from(7), F23E::from(5)]
        );
        assert_eq!(
            evaluation.layers[3],
            vec![
                F23E::from(1),
                F23E::from(6),
                F23E::from(9),
                F23E::from(19),
                F23E::from(17),
                F23E::from(18),
                F23E::from(2),
                F23E::from(3)
            ]
        );
        assert_eq!(evaluation.layers[4], input);
    }

    #[test]
    fn test_gkr_complete_verification() {
        let circuit = thaler_book_circuit().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        let proof = gkr_prove(&circuit, &input).unwrap();
        let result = gkr_verify(&proof, &circuit);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gkr_complete_verification_lambda() {
        let circuit = lambda_post_circuit().unwrap();
        let input = [F23E::from(3), F23E::from(1)];

        let proof = gkr_prove(&circuit, &input).unwrap();
        let result = gkr_verify(&proof, &circuit);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gkr_complete_verification_four_layer() {
        let circuit = four_layer_circuit().unwrap();
        let input: Vec<F23E> = (0u64..16u64).map(F23E::from).collect();

        let proof = gkr_prove(&circuit, &input).unwrap();
        let result = gkr_verify(&proof, &circuit);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gkr_protocol_lambda_invalid_outputs() {
        let circuit = lambda_post_circuit().unwrap();
        let input = [F23E::from(3), F23E::from(1)];

        let proof_result = gkr_prove(&circuit, &input);
        let mut proof = proof_result.unwrap();

        // Corrupt the output values
        proof.output_values = vec![F23E::from(1), F23E::from(2)];

        let verification_result = gkr_verify(&proof, &circuit);

        assert!(
            matches!(verification_result, Ok(false)),
            "The protocol should reject a fake output"
        )
    }

    #[test]
    fn test_invalid_proof_rejection() {
        let circuit = thaler_book_circuit().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        // Generate a valid proof
        let mut proof = gkr_prove(&circuit, &input).expect("Proof generation failed");

        // Corrupt the proof by modifying a round polynomial g_0.
        proof.layer_proofs[0].sumcheck_proof.round_polynomials[0] =
            Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);

        let verification_result = gkr_verify(&proof, &circuit);

        assert!(
            matches!(verification_result, Ok(false)),
            "The protocol should reject an invalid proof"
        )
    }
}
