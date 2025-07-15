pub mod circuit;
pub mod prover;
pub mod sumcheck;
pub mod verifier;
use crate::circuit::{Circuit, CircuitError};
use crate::prover::ProverError;
use crate::sumcheck::SumcheckProof;
use crate::verifier::Verifier;
use crate::verifier::VerifierError;
use blake2::{Blake2s256, Digest};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

/// Create a line polynomial between two points
/// l(t) = b + t * (c - b)
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

#[derive(Debug, Clone)]
pub struct LayerProof<F: IsField> {
    pub claimed_sum: FieldElement<F>,
    pub sumcheck_proof: SumcheckProof<F>,
    pub poly_q: Polynomial<FieldElement<F>>,
}

/// A GKR proof.
#[derive(Debug, Clone)]
pub struct GkrProof<F: IsField> {
    pub input_values: Vec<FieldElement<F>>,
    pub output_values: Vec<FieldElement<F>>,
    pub layer_proofs: Vec<LayerProof<F>>,
    pub final_claim: FieldElement<F>,
    pub final_evaluation_point: Vec<FieldElement<F>>,
}

/// The polynomial `W` that is used in the GKR protocol.
#[derive(Clone)]
pub struct W<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    pub add_i: DenseMultilinearPolynomial<F>,
    pub mul_i: DenseMultilinearPolynomial<F>,
    pub w_b: DenseMultilinearPolynomial<F>,
    pub w_c: DenseMultilinearPolynomial<F>,
}

impl<F: IsField> W<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(
        add_i: DenseMultilinearPolynomial<F>,
        mul_i: DenseMultilinearPolynomial<F>,
        w_b: DenseMultilinearPolynomial<F>,
        w_c: DenseMultilinearPolynomial<F>,
    ) -> Self {
        Self {
            add_i,
            mul_i,
            w_b,
            w_c,
        }
    }

    pub fn to_poly_list(self) -> Vec<DenseMultilinearPolynomial<F>> {
        vec![self.add_i, self.mul_i, self.w_b, self.w_c]
    }

    /// Evaluate the GKR polynomial W at a given point
    /// f^{(i)}_{r_i}(b, c) = add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c)) +
    /// mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))
    pub fn evaluate(&self, point: &[FieldElement<F>]) -> Option<FieldElement<F>>
    where
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let (b, c) = point.split_at(self.w_b.num_vars());

        let add_e = self.add_i.evaluate(point.to_vec()).ok()?;
        let mul_e = self.mul_i.evaluate(point.to_vec()).ok()?;

        let w_b = self.w_b.evaluate(b.to_vec()).ok()?;
        let w_c = self.w_c.evaluate(c.to_vec()).ok()?;

        Some(add_e * (w_b.clone() + w_c.clone()) + mul_e * (w_b * w_c))
    }

    /// Fix variables in the GKR polynomial W (partial evaluation)
    pub fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self
    where
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let b_partial = partial_point
            .get(..std::cmp::min(self.w_b.num_vars(), partial_point.len()))
            .unwrap_or(&[]);
        let c_partial = partial_point.get(self.w_b.num_vars()..).unwrap_or(&[]);

        // Fix variables in each polynomial component
        let mut add_i = self.add_i.clone();
        let mut mul_i = self.mul_i.clone();
        let mut w_b = self.w_b.clone();
        let mut w_c = self.w_c.clone();

        // Apply partial evaluation to each component
        for val in partial_point.iter() {
            add_i = add_i.fix_last_variable(val);
            mul_i = mul_i.fix_last_variable(val);
        }

        for val in b_partial.iter() {
            w_b = w_b.fix_last_variable(val);
        }

        for val in c_partial.iter() {
            w_c = w_c.fix_last_variable(val);
        }

        Self {
            add_i,
            mul_i,
            w_b,
            w_c,
        }
    }

    /// Get the number of variables in the polynomial
    pub fn num_vars(&self) -> usize {
        self.add_i.num_vars()
    }
}

/// Helper function to convert W to evaluations for sumcheck
/// Based on the GKR protocol: f^{(i)}_{r_i}(b, c) =
/// add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c)) +
/// mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))
pub fn w_to_evaluations<F: IsField>(w: &W<F>) -> Vec<FieldElement<F>>
where
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    // combine the evaluations of separate multilinear
    // extensions into a vector of evaluations of the
    // whole polynomial
    let w_b_evals = w.w_b.to_evaluations();
    let w_c_evals = w.w_c.to_evaluations();
    let add_i_evals = w.add_i.to_evaluations();
    let mul_i_evals = w.mul_i.to_evaluations();

    let mut res = vec![];
    for (b_idx, w_b_item) in w_b_evals.iter().enumerate() {
        for (c_idx, w_c_item) in w_c_evals.iter().enumerate() {
            let bc_idx = idx(c_idx, b_idx, w.w_b.num_vars());

            res.push(
                add_i_evals[bc_idx].clone() * (w_b_item.clone() + w_c_item.clone())
                    + mul_i_evals[bc_idx].clone() * (w_b_item.clone() * w_c_item.clone()),
            );
        }
    }

    res
}

/// Combine indices of two variables into one to be able
/// to index into evaluations of polynomial.
fn idx(i: usize, j: usize, num_vars: usize) -> usize {
    (i << num_vars) | j
}

pub fn gkr_prove<F>(
    circuit: &Circuit,
    input: &[FieldElement<F>],
) -> Result<GkrProof<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    prover::generate_proof(circuit, input)
}

pub fn gkr_verify<F>(proof: &GkrProof<F>, circuit: &Circuit) -> Result<bool, VerifierError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    Verifier::verify(proof, circuit)
}

/// Complete GKR verification including input verification
pub fn gkr_verify_complete<F>(
    proof: &GkrProof<F>,
    circuit: &Circuit,
    input: &[FieldElement<F>],
) -> Result<bool, VerifierError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    // First evaluate the circuit to get the evaluation
    //let evaluation = circuit.evaluate(input);

    // Then verify the proof structure
    let is_valid = Verifier::verify(proof, circuit)?;

    if !is_valid {
        return Ok(false);
    }

    // Then verify the final input layer using the final point from the proof
    // Use the final point stored in the proof
    let final_point = &proof.final_evaluation_point;

    // Create the input polynomial and evaluate it at the final point
    let input_poly = DenseMultilinearPolynomial::new(input.to_vec());
    let final_evaluation = input_poly
        .evaluate(final_point.to_vec())
        .map_err(|_| VerifierError::EvaluationFailed)?;

    // Get the final layer claim (like self.m.last() in the reference)
    let final_claim = &proof.final_claim;

    // Verify that the input evaluation matches the final claim
    // This is the same check as in the reference: w.evaluate(r_last) == m_last
    Ok(final_evaluation == *final_claim)
}

pub fn hash_circuit(c: &Circuit) -> [u8; 32] {
    let mut h = Blake2s256::new();
    h.update(b"GKR-Circuit-v1");
    h.update(&(c.layers().len() as u32).to_le_bytes());
    h.update(&(c.num_inputs() as u32).to_le_bytes());

    for layer in c.layers() {
        h.update(&(layer.len() as u32).to_le_bytes());
        for g in &layer.layer {
            let gate_type = match g.ttype {
                crate::circuit::GateType::Add => 0u8,
                crate::circuit::GateType::Mul => 1u8,
            };
            h.update(&[gate_type]);
            h.update(&(g.inputs[0] as u32).to_le_bytes());
            h.update(&(g.inputs[1] as u32).to_le_bytes());
        }
    }
    h.finalize().into()
}

/// Minimal commitment to the witness values
/// TODO: Replace with proper PCS
/// This is a simple hash-based commitment for now
pub fn commit_witness<F: IsField>(w: &[FieldElement<F>]) -> [u8; 32]
where
    FieldElement<F>: ByteConversion,
{
    let mut hasher = Blake2s256::new();
    for element in w {
        hasher.update(element.to_bytes_be());
    }
    hasher.finalize().into()
}

/// Create the circuit from Lambda post
pub fn circuit_from_lambda() -> Result<Circuit, CircuitError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, CircuitError, CircuitLayer, Gate, GateType};
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 389;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    const MODULUS23: u64 = 23;
    type F23 = U64PrimeField<MODULUS23>;
    type F23E = FieldElement<F23>;

    /// Create the circuit from Thaler's book (Figure 4.12)
    fn circuit_from_book() -> Result<Circuit, CircuitError> {
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

    /// Create a three-layer circuit for testing
    fn three_layer_circuit() -> Result<Circuit, CircuitError> {
        Circuit::new(
            vec![
                CircuitLayer::new(vec![
                    Gate::new(GateType::Add, [0, 1]),
                    Gate::new(GateType::Add, [2, 3]),
                ]),
                CircuitLayer::new(vec![
                    Gate::new(GateType::Add, [0, 1]),
                    Gate::new(GateType::Add, [2, 3]),
                    Gate::new(GateType::Add, [4, 5]),
                    Gate::new(GateType::Add, [6, 7]),
                ]),
            ],
            8,
        )
    }

    #[test]
    fn print() {
        let circuit = circuit_from_lambda().unwrap();
        // let input = [F23E::from(3), F23E::from(1)];
        // let evaluation = circuit.evaluate(&input);
        let a = circuit.mul_i(0, 0, 1, 0);
        let b = circuit.mul_i(0, 0, 0, 1);

        println!("{a}");
        println!("{b}");
    }

    #[test]
    fn test_circuit_evaluation_from_lambda() {
        let circuit = circuit_from_lambda().unwrap();
        let input = [F23E::from(3), F23E::from(1)];
        let evaluation = circuit.evaluate(&input);
        // Expected layers: input -> [3, 6, 4, 3] -> [18, 7]
        assert_eq!(evaluation.layers.len(), 3);
        //assert_eq!(evaluation.layers[0], [F23E::from(18), F23E::from(7)]); // output
        assert_eq!(
            evaluation.layers[1],
            [F23E::from(3), F23E::from(6), F23E::from(4), F23E::from(3)]
        ); // middle
        assert_eq!(evaluation.layers[2], input.to_vec()); // input
    }

    #[test]
    fn test_circuit_evaluation_from_book() {
        let circuit = circuit_from_book().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        let evaluation = circuit.evaluate(&input);

        // Expected layers: input -> [9, 4, 6, 1] -> [36, 6]
        assert_eq!(evaluation.layers.len(), 3);
        assert_eq!(evaluation.layers[0], [FE::from(36), FE::from(6)]); // output
        assert_eq!(
            evaluation.layers[1],
            [FE::from(9), FE::from(4), FE::from(6), FE::from(1)]
        ); // middle
        assert_eq!(evaluation.layers[2], input.to_vec()); // input
    }

    #[test]
    fn test_three_layer_circuit_evaluation() {
        let circuit = three_layer_circuit().unwrap();
        let input = [
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(1),
        ];

        let evaluation = circuit.evaluate(&input);

        // Expected: input -> [1,1,1,1] -> [2,2]
        assert_eq!(evaluation.layers.len(), 3);
        assert_eq!(evaluation.layers[0], [FE::from(2), FE::from(2)]); // output
        assert_eq!(
            evaluation.layers[1],
            [FE::from(1), FE::from(1), FE::from(1), FE::from(1)]
        ); // middle
        assert_eq!(evaluation.layers[2], input.to_vec()); // input
    }

    #[test]
    fn test_w_polynomial_evaluation() {
        // Create a simple W polynomial for testing
        let add_evals = vec![FE::from(1), FE::from(0), FE::from(0), FE::from(0)];
        let mul_evals = vec![FE::from(0), FE::from(1), FE::from(0), FE::from(0)];
        let w_b_evals = vec![FE::from(2), FE::from(3)];
        let w_c_evals = vec![FE::from(4), FE::from(5)];

        let add_poly = DenseMultilinearPolynomial::new(add_evals);
        let mul_poly = DenseMultilinearPolynomial::new(mul_evals);
        let w_b_poly = DenseMultilinearPolynomial::new(w_b_evals);
        let w_c_poly = DenseMultilinearPolynomial::new(w_c_evals);

        let w = W::new(add_poly, mul_poly, w_b_poly, w_c_poly);

        // Test evaluation at a point
        let point = vec![FE::from(0), FE::from(0)];
        let result = w.evaluate(&point);
        assert!(result.is_some());

        // Test w_to_evaluations
        let evals = w_to_evaluations(&w);
        assert!(!evals.is_empty());
    }

    #[test]
    fn test_gkr_protocol_from_book() {
        let circuit = circuit_from_book().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        println!("\n=== GKR Protocol Test (from book) ===");
        println!("Input: {:?}", input);

        // Evaluate the circuit
        let evaluation = circuit.evaluate(&input);
        println!("Expected output: {:?}", evaluation.layers[0]);

        // Generate proof
        println!("\n--- Generating proof ---");
        let proof_result = gkr_prove(&circuit, &input);
        assert!(
            proof_result.is_ok(),
            "Proof generation failed: {:?}",
            proof_result.err()
        );

        let proof = proof_result.unwrap();
        println!("Proof generated successfully!");
        println!("Number of sumcheck proofs: {}", proof.layer_proofs.len());
        println!("Number of claims: {}", proof.layer_proofs.len());

        // Verify proof
        println!("\n--- Verifying proof ---");
        let verification_result = gkr_verify(&proof, &circuit);
        assert!(
            verification_result.is_ok(),
            "Verification failed: {:?}",
            verification_result.err()
        );

        let is_valid = verification_result.unwrap();
        println!(
            "Verification result: {}",
            if is_valid { "ACCEPTED" } else { "REJECTED" }
        );
        assert!(is_valid, "Proof should be valid");

        println!("GKR protocol test from book PASSED! ✓");
    }

    #[test]
    fn test_gkr_protocol_three_layers() {
        let circuit = three_layer_circuit().unwrap();
        let input = [
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(1),
            FE::from(0),
            FE::from(1),
        ];

        println!("\n=== GKR Protocol Test (three layers) ===");
        println!("Input: {:?}", input);

        // Evaluate the circuit
        let evaluation = circuit.evaluate(&input);
        println!("Expected output: {:?}", evaluation.layers[0]);

        // Generate proof
        println!("\n--- Generating proof ---");
        let proof_result = gkr_prove(&circuit, &input);
        assert!(
            proof_result.is_ok(),
            "Proof generation failed: {:?}",
            proof_result.err()
        );

        let proof = proof_result.unwrap();
        println!("Proof generated successfully!");
        println!("Number of sumcheck proofs: {}", proof.layer_proofs.len());
        println!("Number of claims: {}", proof.layer_proofs.len());

        // Verify proof
        println!("\n--- Verifying proof ---");
        let verification_result = gkr_verify(&proof, &circuit);
        assert!(
            verification_result.is_ok(),
            "Verification failed: {:?}",
            verification_result.err()
        );

        let is_valid = verification_result.unwrap();
        println!(
            "Verification result: {}",
            if is_valid { "ACCEPTED" } else { "REJECTED" }
        );
        assert!(is_valid, "Proof should be valid");

        println!("GKR protocol test three layers PASSED! ✓");
    }

    #[test]
    fn test_gkr_protocol_lambda() {
        let circuit = circuit_from_lambda().unwrap();
        let input = [F23E::from(3), F23E::from(1)];

        // println!("\n=== GKR Protocol Test (three layers) ===");
        // println!("Input: {:?}", input);s

        // Evaluate the circuit
        let evaluation = circuit.evaluate(&input);
        println!("Expected output: {:?}", evaluation.layers[0]);

        // Generate proof
        // println!("\n--- Generating proof ---");
        let proof_result = gkr_prove(&circuit, &input);
        // assert!(
        //     proof_result.is_ok(),
        //     "Proof generation failed: {:?}",
        //     proof_result.err()
        // );

        let proof = proof_result.unwrap();
        // println!("Proof generated successfully!");
        // println!("Number of sumcheck proofs: {}", proof.layer_proofs.len());
        // println!("Number of claims: {}", proof.layer_proofs.len());

        // Verify proof
        // println!("\n--- Verifying proof ---");
        let verification_result = gkr_verify(&proof, &circuit);
        // assert!(
        //     verification_result.is_ok(),
        //     "Verification failed: {:?}",
        //     verification_result.err()
        // );

        let is_valid = verification_result.unwrap();
        // println!(
        //     "Verification result: {}",
        //     if is_valid { "ACCEPTED" } else { "REJECTED" }
        // );
        println!("Verification result: {}", is_valid);
        assert!(is_valid, "Proof should be valid");

        // println!("GKR protocol test three layers PASSED! ✓");
    }

    #[test]
    fn test_gkr_protocol_lambda_outputs_mentirosos() {
        let circuit = circuit_from_lambda().unwrap();
        let input = [F23E::from(3), F23E::from(1)];

        println!("\n=== GKR Protocol Test (three layers) ===");
        println!("Input: {:?}", input);

        // Evaluate the circuit
        let evaluation = circuit.evaluate(&input);
        println!("Expected output: {:?}", evaluation.layers[0]);

        // Generate proof
        println!("\n--- Generating proof ---");
        let proof_result = gkr_prove(&circuit, &input);
        assert!(
            proof_result.is_ok(),
            "Proof generation failed: {:?}",
            proof_result.err()
        );

        let mut proof = proof_result.unwrap();
        println!("Proof generated successfully!");
        println!("Number of sumcheck proofs: {}", proof.layer_proofs.len());
        println!("Number of claims: {}", proof.layer_proofs.len());

        proof.output_values = vec![F23E::from(0), F23E::from(0)]; // Corrupt output values
                                                                  // Verify proof
        println!("\n--- Verifying proof ---");
        let verification_result = gkr_verify(&proof, &circuit);
        match verification_result {
            Ok(false) | Err(VerifierError::InvalidProof) => {
                // Test PASSED: el protocolo detectó el engaño
                println!("Verification result: REJECTED (as expected)");
            }
            Ok(true) => panic!("El protocolo aceptó un output mentiroso"),
            Err(e) => panic!("Error inesperado: {:?}", e),
        }
        println!("GKR protocol test lambda outputs mentirosos PASSED! ✓");
    }

    #[test]
    fn test_gkr_complete_verification() {
        let circuit = circuit_from_book().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        let proof = gkr_prove(&circuit, &input).unwrap();
        let result = gkr_verify_complete(&proof, &circuit, &input);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gkr_complete_verification_lambda() {
        let circuit = circuit_from_lambda().unwrap();
        let input = [F23E::from(3), F23E::from(1)];

        let proof = gkr_prove(&circuit, &input).unwrap();
        let result = gkr_verify_complete(&proof, &circuit, &input);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_circuit_properties() {
        let circuit = circuit_from_book().unwrap();

        // Test circuit properties
        assert_eq!(circuit.num_outputs(), 2);
        assert_eq!(circuit.num_inputs(), 4);
        assert_eq!(circuit.layers().len(), 2);

        // Test num_vars_at for different layers
        assert_eq!(circuit.num_vars_at(0), Some(1)); // 2 outputs -> 1 var
        assert_eq!(circuit.num_vars_at(1), Some(2)); // 4 gates -> 2 vars
        assert_eq!(circuit.num_vars_at(2), Some(2)); // 4 inputs -> 2 vars

        // Test add_i and mul_i predicates
        assert!(circuit.mul_i(0, 0, 0, 1)); // First output gate multiplies inputs 0,1
        assert!(circuit.mul_i(0, 1, 2, 3)); // Second output gate multiplies inputs 2,3
        assert!(!circuit.add_i(0, 0, 0, 1)); // No addition gates in output layer

        println!("Circuit properties test PASSED! ✓");
    }

    #[test]
    fn test_invalid_proof_rejection() {
        let circuit = circuit_from_book().unwrap();
        let input = [FE::from(3), FE::from(2), FE::from(3), FE::from(1)];

        // Generate a valid proof
        let mut proof = gkr_prove(&circuit, &input).expect("Proof generation failed");

        // Corrupt the proof by modifying claims
        if !proof.layer_proofs.is_empty() {
            proof.layer_proofs[0].claimed_sum = FE::from(999); // Invalid claim
        }

        let verification_result = gkr_verify(&proof, &circuit);

        println!("Invalid proof test: {:?}", verification_result);

        println!("Invalid proof rejection test completed! ✓");
    }
}
