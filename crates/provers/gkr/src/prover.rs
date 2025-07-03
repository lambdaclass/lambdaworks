use crate::circuit::{Circuit, CircuitEvaluation};
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_sumcheck::Channel;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("Verification failed")]
    VerificationFailed,
    #[error("Circuit evaluation failed")]
    CircuitEvaluationFailed,
    #[error("Sumcheck proof generation failed")]
    SumcheckFailed,
}

/// Builds the GKR polynomial for a given layer `i` by combining the wiring predicates
/// with the evaluations of the next layer.
///
/// The GKR polynomial is defined as:
/// f~_{r_i}(b, c) = add~(r_i, b, c) * (W~_{i+1}(b) + W~_{i+1}(c)) + mul~(r_i, b, c) * (W~_{i+1}(b) * W~_{i+1}(c))
pub(crate) fn build_gkr_polynomial<F: IsField>(
    circuit: &Circuit,
    r_i: &[FieldElement<F>], // The random fixed values for the variable 'a'.
    // e.g. In the post: i = 0. The gates are 0 and 1, then 'a' in F^1.
    // i = 2. The gates are 00, 01, 10, 11. Then 'a' in F^2.
    evaluation: &CircuitEvaluation<FieldElement<F>>,
    layer_idx: usize,
) -> DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    // Get the multilinear extensions of the wiring predicates fixed at r_i
    let add_i_poly = circuit.add_i_ext::<F>(r_i, layer_idx);
    let mul_i_poly = circuit.mul_i_ext::<F>(r_i, layer_idx);
    // QUESTION: Is it necessary x_next_poly. cant be directly w_next_evals?
    //let w_next_poly = DenseMultilinearPolynomial::new(evaluation.layers[layer_idx + 1].clone());

    let add_i_evals = add_i_poly.to_evaluations();
    let mul_i_evals = mul_i_poly.to_evaluations();

    let w_next_evals = evaluation.layers[layer_idx + 1].clone();
    //let w_next_evals = w_next_poly.to_evaluations();

    let num_vars_next = circuit.num_vars_at(layer_idx + 1).unwrap_or(0);
    let mut gkr_poly_evals = Vec::with_capacity(1 << (2 * num_vars_next)); // 2^{2*k_{i+1}} because to build the DenseMultilinearPolynomial, we need the evaluations of f at b and c, each of them at the hypercube of the next layer.

    // Construct the GKR polynomial evaluations directly
    for c_idx in 0..(1 << num_vars_next) {
        // 2^{k_{i+1}}. (00, ..., 11) = (0, ..., 3). 00
        for b_idx in 0..(1 << num_vars_next) {
            // 01
            let bc_idx = c_idx + (b_idx << num_vars_next); // 0001
            let w_b: &FieldElement<F> = &w_next_evals[b_idx];
            let w_c = &w_next_evals[c_idx];
            let gkr_eval = &add_i_evals[bc_idx] * (w_b + w_c) + &mul_i_evals[bc_idx] * (w_b * w_c);
            gkr_poly_evals.push(gkr_eval);
        }
    }

    DenseMultilinearPolynomial::new(gkr_poly_evals)
}

/// GKR-specific sumcheck prover that implements the protocol from scratch
fn gkr_sumcheck_prove<F, T>(
    factors: Vec<DenseMultilinearPolynomial<F>>,
    transcript: &mut T,
) -> Result<(FieldElement<F>, Vec<Polynomial<FieldElement<F>>>), ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync + Copy,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F> + Channel<F>,
{
    if factors.is_empty() {
        return Err(ProverError::SumcheckFailed);
    }

    let num_vars = factors[0].num_vars();
    if factors.iter().any(|p| p.num_vars() != num_vars) {
        return Err(ProverError::SumcheckFailed);
    }

    // Compute the initial claimed sum by evaluating the product over all points
    let mut claimed_sum = FieldElement::zero();
    for point in 0..(1 << num_vars) {
        let mut point_vec = Vec::with_capacity(num_vars);
        for i in 0..num_vars {
            let bit = (point >> i) & 1;
            point_vec.push(FieldElement::from(bit as u64));
        }

        let mut product = FieldElement::one();
        for factor in &factors {
            let eval = factor
                .evaluate(point_vec.clone())
                .map_err(|_| ProverError::SumcheckFailed)?;
            product = product * eval;
        }
        claimed_sum = claimed_sum + product;
    }

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);

    // Execute rounds
    for j in 0..num_vars {
        // Compute the round polynomial g_j by interpolation
        let num_eval_points = factors.len() + 1;
        let mut evaluation_points_x = Vec::with_capacity(num_eval_points);
        let mut evaluations_y = Vec::with_capacity(num_eval_points);

        // Prefix for evaluation points: (r1, r2, ..., r_{j-1}, eval_point_x)
        let mut current_point_prefix = challenges.clone();
        current_point_prefix.push(FieldElement::zero());

        for i in 0..num_eval_points {
            let eval_point_x = FieldElement::from(i as u64);
            evaluation_points_x.push(eval_point_x.clone());

            // Set the actual value for X_j in the prefix
            *current_point_prefix.last_mut().unwrap() = eval_point_x;

            // Compute g_j(eval_point_x) = sum over remaining variables
            let mut g_j_at_eval_point = FieldElement::zero();
            for remaining_point in 0..(1 << (num_vars - j - 1)) {
                let mut full_point = current_point_prefix.clone();
                for k in 0..(num_vars - j - 1) {
                    let bit = (remaining_point >> k) & 1;
                    full_point.push(FieldElement::from(bit as u64));
                }

                let mut product = FieldElement::one();
                for factor in &factors {
                    let eval = factor
                        .evaluate(full_point.clone())
                        .map_err(|_| ProverError::SumcheckFailed)?;
                    product = product * eval;
                }
                g_j_at_eval_point = g_j_at_eval_point + product;
            }
            evaluations_y.push(g_j_at_eval_point);
        }

        let g_j = Polynomial::interpolate(&evaluation_points_x, &evaluations_y)
            .map_err(|_| ProverError::SumcheckFailed)?;

        // Debug: Print the polynomial coefficients
        println!(
            "Sumcheck round {}: g_j coefficients: {:?}",
            j,
            g_j.coefficients()
        );
        println!(
            "Sumcheck round {}: g_j(0) = {:?}, g_j(1) = {:?}",
            j,
            g_j.evaluate(&FieldElement::<F>::zero()),
            g_j.evaluate(&FieldElement::<F>::one())
        );

        proof_polys.push(g_j);

        // Generate challenge for the next round (if not the last round)
        if j < num_vars - 1 {
            let challenge = transcript.sample_field_element();
            println!("Sumcheck round {}: Generated challenge {:?}", j, challenge);
            challenges.push(challenge);
        }
    }

    Ok((claimed_sum, proof_polys))
}

/// Generate a GKR proof
/// This implements the prover side of the GKR protocol
/// TODO: generate_proof no tendría que tener como input evaluation sino los inputs del circuito y que el prover calcule las evaluaciones.
pub fn generate_proof<F>(
    circuit: &Circuit,
    input: &[FieldElement<F>],
) -> Result<crate::Proof<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    let mut sumcheck_proofs = vec![];

    let mut claims_phase2 = vec![];
    //let mut layer_commitments = vec![];
    let mut layer_claims = vec![];

    // Evaluate the circuit on the given input.
    let evaluation = circuit.evaluate(input);

    // Generate commitments to layer evaluations
    // TODO: cambiar esto por mandar al transcript algo que dependa del circuito, los inputs y los outputs.
    // Lo que está ahora no sirve pq el verfiier no tiene acceso a las evaluaciones.
    // https://eprint.iacr.org/2025/118.pdf pag 7 (2.1) y 8 (2.2)

    // Commitment part
    // Acording to the paper this is ...

    // for layer_evals in &evaluation.layers {
    //     // Simple commitment: sum of all evaluations
    //     let mut commitment = FieldElement::zero();
    //     for eval in layer_evals {
    //         commitment = commitment + eval.clone();
    //     }
    //     layer_commitments.push(commitment.clone());
    //     transcript.append_bytes(&commitment.to_bytes_be());
    // }

    // r = H(⟨C⟩, x, y)

    let mut transcript = DefaultTranscript::<F>::default();

    // 0. Append the circuit data to the transcript.
    transcript.append_bytes(&(circuit.layers().len() as u32).to_le_bytes());
    transcript.append_bytes(&(circuit.num_inputs() as u32).to_le_bytes()); // QUESTION: Is it necessary to append num_inputs?
                                                                           // For each layer and each gate, append the gate's type and input indeces.
    for layer in circuit.layers() {
        transcript.append_bytes(&(layer.len() as u32).to_le_bytes());
        for gate in &layer.layer {
            let gate_type = match gate.ttype {
                crate::circuit::GateType::Add => 0u8,
                crate::circuit::GateType::Mul => 1u8,
            };
            transcript.append_bytes(&[gate_type]);
            transcript.append_bytes(&(gate.inputs[0] as u32).to_le_bytes());
            transcript.append_bytes(&(gate.inputs[1] as u32).to_le_bytes());
        }
    }

    // 1. x public inputs (last layer of evaluation)
    for x in input {
        transcript.append_bytes(&x.to_bytes_be());
    }

    // 2. y outputs (first layer of evaluation)
    for y in &evaluation.layers[0] {
        transcript.append_bytes(&y.to_bytes_be());
    }

    // Get the number of variables for the output layer
    // TODO: sacar el unwrap. num_vars queremos que este como parte del struct del layer.
    let k_0 = circuit.num_vars_at(0).unwrap_or(0);
    let mut r_i: Vec<FieldElement<F>> = (0..k_0)
        .map(|_| transcript.sample_field_element())
        .collect();

    // For each layer, run the GKR protocol
    for i in 0..circuit.layers().len() {
        let gkr_poly = build_gkr_polynomial(circuit, &r_i, &evaluation, i);
        let w_next_poly = DenseMultilinearPolynomial::new(evaluation.layers[i + 1].clone());

        // Run sumcheck on the GKR polynomial
        // TODO: más documentación. que es sum. hacer referencia al post.
        // Suponemos que proof tiene los polinomios g_i que va mandando el prover al verifier en el sumcheck protocol en nuestro post.
        //let (sum, proof) = prove(vec![gkr_poly]).map_err(|_| ProverError::SumcheckFailed)?;
        println!(
            "GKR Layer {}: Starting sumcheck with transcript state: {:?}",
            i,
            transcript.state()
        );
        println!("GKR Layer {}: Using challenges r_i: {:?}", i, r_i);
        let (sum, proof) = gkr_sumcheck_prove(vec![gkr_poly], &mut transcript)
            .map_err(|_| ProverError::SumcheckFailed)?;
        println!(
            "GKR Layer {}: Finished sumcheck with transcript state: {:?}",
            i,
            transcript.state()
        );

        // Update transcript and store results
        println!("GKR Layer {}: Adding sum result {:?} to transcript", i, sum);
        transcript.append_bytes(&sum.to_bytes_be());
        claims_phase2.push(sum);
        sumcheck_proofs.push(proof);

        println!(
            "GKR Layer {}: After sumcheck, transcript state: {:?}",
            i,
            transcript.state()
        );

        // Sample challenges for the next round
        let k_next = circuit.num_vars_at(i + 1).unwrap_or(0);
        let num_sumcheck_challenges = 2 * k_next;

        // (s_1, ..., s_{2k})
        let sumcheck_challenges: Vec<FieldElement<F>> = (0..num_sumcheck_challenges)
            .map(|_| transcript.sample_field_element())
            .collect();

        // r* in the post
        let r_last = transcript.sample_field_element();

        println!(
            "GKR Layer {}: After sampling challenges, transcript state: {:?}",
            i,
            transcript.state()
        );
        println!(
            "GKR Layer {}: Generated challenges: sumcheck_challenges={:?}, r_last={:?}",
            i, sumcheck_challenges, r_last
        );

        // Construct the next round's random point
        let (b, c) = sumcheck_challenges.split_at(k_next);
        let r_i_next = crate::line(b, c, &r_last);

        println!("GKR Layer {}: Constructed r_i_next: {:?}", i, r_i_next);

        // Evaluate W_{i+1} at the new point and add to transcript
        // TODO: cambiar. El verifier no conoce w_next_poly como para apendearlo al transcript.
        if let Ok(next_claim) = w_next_poly.evaluate(r_i_next.clone()) {
            println!(
                "GKR Layer {}: Evaluated W_{} at r_i_next = {:?}, got claim: {:?}",
                i,
                i + 1,
                r_i_next,
                next_claim
            );
            transcript.append_bytes(&next_claim.to_bytes_be());
            layer_claims.push(next_claim);
            println!(
                "GKR Layer {}: After adding layer claim, transcript state: {:?}",
                i,
                transcript.state()
            );
        }

        r_i = r_i_next;
    }

    // Store the final point for input verification
    // The final point should have the dimension of the input layer
    let num_input_vars = (evaluation.layers.last().unwrap().len() as f64).log2() as usize;
    let final_point = if r_i.len() >= num_input_vars {
        r_i[..num_input_vars].to_vec()
    } else {
        // If r_i is smaller than needed, pad with zeros
        let mut padded = r_i.clone();
        while padded.len() < num_input_vars {
            padded.push(FieldElement::zero());
        }
        padded
    };

    // Evaluate the input layer at the final point to get the correct final claim
    let input_poly = DenseMultilinearPolynomial::new(evaluation.layers.last().unwrap().clone());
    let final_claim = input_poly
        .evaluate(final_point.clone())
        .unwrap_or_else(|_| FieldElement::zero());

    // Add the final claim to layer_claims (this is like m.last() in the reference)
    layer_claims.push(final_claim);

    Ok(crate::Proof {
        sumcheck_proofs,
        claims_phase2,
        //layer_commitments,
        //witness_comm: alpha.to_vec(),
        //line_polys,
        final_point,
        layer_claims,
    })
}
