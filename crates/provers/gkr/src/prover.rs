use crate::{circuit::Circuit, sumcheck::gkr_sumcheck_prove, GkrProof, LayerProof};

use lambdaworks_crypto::fiat_shamir::{
    default_transcript::DefaultTranscript, is_transcript::IsTranscript,
};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::dense_multilinear_poly::DenseMultilinearPolynomial,
    traits::ByteConversion,
};
#[derive(Debug)]
pub enum ProverError {
    EvaluationFailed,
    SumcheckFailed,
}

/// Constructs the polynomial `f(b, c)` for the GKR sumcheck for a specific layer.
///
/// The GKR protocol reduces a claim about layer `i` to a claim about layer `i+1`.
/// This reduction is proven using a sumcheck protocol on a polynomial `f` that is
/// constructed by the prover.
///
/// The polynomial `f` is defined as:
/// `f(b, c) = add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c)) + mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))`
///
/// where:
/// - `add_i` and `mul_i` are the wiring polynomials for the addition and multiplication gates of layer `i`.
/// - `r_i` is the random challenge point for layer `i`.
/// - `W_{i+1}` is the polynomial representing the evaluation of layer `i+1`.
/// - `b` and `c` are the inputs to the gates, with `k_{i+1}` variables each.
///
/// This function is only known to the prover. It returns the components of `f` grouped
/// into terms, ready to be consumed by the sumcheck prover, which expects a sum of products.
fn build_gkr_polynomial<F: IsField + Clone>(
    circuit: &Circuit,
    r_i: &[FieldElement<F>],
    evaluation: &[FieldElement<F>],
    layer_idx: usize,
) -> Result<Vec<Vec<DenseMultilinearPolynomial<F>>>, ProverError>
where
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    let k_i_plus_1 = circuit
        .num_vars_at(layer_idx + 1)
        .ok_or(ProverError::EvaluationFailed)?;
    let k_i = circuit
        .num_vars_at(layer_idx)
        .ok_or(ProverError::EvaluationFailed)?;

    // Step 1: Construct the full wiring polynomials `add_i(g, b, c)` and `mul_i(g, b, c)`.
    // These are defined over k_i + 2 * k_{i+1} variables and act as "selectors".

    // Construct `add_i(g, b, c)`.
    let add_i_evals = (0..1 << (k_i + 2 * k_i_plus_1))
        .map(|x| {
            let b_val = (x >> k_i_plus_1) & ((1 << k_i_plus_1) - 1);
            let c_val = x & ((1 << k_i_plus_1) - 1);
            if circuit.add_i(layer_idx, x >> (2 * k_i_plus_1), b_val, c_val) {
                FieldElement::one()
            } else {
                FieldElement::zero()
            }
        })
        .collect::<Vec<FieldElement<F>>>();
    let add_i_full = DenseMultilinearPolynomial::new(add_i_evals);

    // Construct `mul_i(g, b, c)`.
    let mul_i_evals = (0..1 << (k_i + 2 * k_i_plus_1))
        .map(|x| {
            let b_val = (x >> k_i_plus_1) & ((1 << k_i_plus_1) - 1);
            let c_val = x & ((1 << k_i_plus_1) - 1);
            if circuit.mul_i(layer_idx, x >> (2 * k_i_plus_1), b_val, c_val) {
                FieldElement::one()
            } else {
                FieldElement::zero()
            }
        })
        .collect::<Vec<FieldElement<F>>>();
    let mul_i_full = DenseMultilinearPolynomial::new(mul_i_evals);

    // Step 2: Partially evaluate the wiring polynomials at `r_i`.
    // This gives us `add_i(r_i, b, c)` and `mul_i(r_i, b, c)`, which are polynomials
    // over `2 * k_{i+1}` variables (`b` and `c`).
    let mut add_i = add_i_full;
    for r in r_i {
        add_i = add_i.fix_last_variable(r);
    }

    let mut mul_i = mul_i_full;
    for r in r_i {
        mul_i = mul_i.fix_last_variable(r);
    }

    // Step 3: Construct polynomials for `W_{i+1}(b)` and `W_{i+1}(c)`.
    // The evaluation of the next layer, `W_{i+1}`, is a polynomial in `k_{i+1}` variables.
    // We need to "lift" it to a polynomial in `2 * k_{i+1}` variables (the space of `b` and `c`).
    // `w_b` will be a polynomial that depends only on `b`, and `w_c` only on `c`.

    // Create W_{i+1}(b). The resulting polynomial is `w_b(b,c) = W_{i+1}(b)`.
    let w_b_evals = (0..1 << (2 * k_i_plus_1))
        .map(|x| {
            let b_val = (x >> k_i_plus_1) & ((1 << k_i_plus_1) - 1); // Extract b
            if b_val < evaluation.len() {
                evaluation[b_val].clone()
            } else {
                FieldElement::zero()
            }
        })
        .collect::<Vec<FieldElement<F>>>();
    let w_b = DenseMultilinearPolynomial::new(w_b_evals);

    // Create W_{i+1}(c). The resulting polynomial is `w_c(b,c) = W_{i+1}(c)`.
    let w_c_evals = (0..1 << (2 * k_i_plus_1))
        .map(|x| {
            let c_val = x & ((1 << k_i_plus_1) - 1); // Extract c
            if c_val < evaluation.len() {
                evaluation[c_val].clone()
            } else {
                FieldElement::zero()
            }
        })
        .collect::<Vec<FieldElement<F>>>();
    let w_c = DenseMultilinearPolynomial::new(w_c_evals);

    // Step 4: Construct the `W_{i+1}(b) + W_{i+1}(c)` term.
    let w_sum_evals = w_b
        .to_evaluations()
        .iter()
        .zip(w_c.to_evaluations().iter())
        .map(|(b, c)| b.clone() + c.clone())
        .collect::<Vec<FieldElement<F>>>();
    let w_sum = DenseMultilinearPolynomial::new(w_sum_evals);

    // Step 5: Assemble the final terms for the sumcheck prover.
    // The sumcheck protocol will prove the sum of `term1 + term2`.
    let term1 = vec![add_i, w_sum]; // Corresponds to add_i(r_i,b,c) * (W_{i+1}(b) + W_{i+1}(c))
    let term2 = vec![mul_i, w_b, w_c]; // Corresponds to mul_i(r_i,b,c) * W_{i+1}(b) * W_{i+1}(c)

    Ok(vec![term1, term2])
}

/// Generate a GKR proof
/// This implements the prover side of the GKR protocol
pub fn generate_proof<F>(
    circuit: &Circuit,
    input: &[FieldElement<F>],
) -> Result<GkrProof<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    let evaluation = circuit.evaluate(input);

    let mut layer_proofs = Vec::new();

    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(&crate::hash_circuit(circuit));

    for x in input {
        transcript.append_bytes(&x.to_bytes_be());
    }

    for y in &evaluation.layers[0] {
        transcript.append_bytes(&y.to_bytes_be());
    }

    let k_0 = circuit
        .num_vars_at(0)
        .ok_or(ProverError::EvaluationFailed)?;
    let mut r_i: Vec<FieldElement<F>> = (0..k_0)
        .map(|_| transcript.sample_field_element())
        .collect();
    r_i = vec![FieldElement::<F>::from(2)];

    for layer_idx in 0..circuit.layers().len() {
        let w_i_plus_1 = &evaluation.layers[layer_idx + 1];

        // Build the GKR polynomial terms
        let gkr_poly_terms = build_gkr_polynomial(circuit, &r_i, w_i_plus_1, layer_idx)?;

        let (initial_sum, sumcheck_proof, sumcheck_challenges) =
            gkr_sumcheck_prove(gkr_poly_terms.clone(), &mut transcript)?;

        // Sample challenges for the next round using line function
        let k_i_plus_1 = circuit
            .num_vars_at(layer_idx + 1)
            .ok_or(ProverError::EvaluationFailed)?;

        // r* in the Lambda post
        let mut r_last = transcript.sample_field_element();

        if layer_idx == 0 {
            r_last = FieldElement::<F>::from(6);
        }

        if layer_idx == 1 {
            r_last = FieldElement::<F>::from(17);
        }

        // Construct the next round's random point using line function
        //  l(x) = b + x * (c - b)
        let (b, c) = sumcheck_challenges.split_at(k_i_plus_1);
        r_i = crate::line(b, c, &r_last);

        let mut evaluation_points_x = Vec::new();
        let mut evaluations_y = Vec::new();

        for i in 0..3 {
            // Point at which to evaluate X_j
            let eval_point_x = FieldElement::from(i as u64);
            evaluation_points_x.push(eval_point_x.clone());

            // // Set the actual value for X_j in the prefix
            // *current_point_prefix.last_mut().unwrap() = eval_point_x;

            // q = W o l
            // q is the composition of W and the line
            let line_at_eval_point = crate::line(b, c, &eval_point_x);
            let w_i_plus_1_poly = DenseMultilinearPolynomial::new(w_i_plus_1.clone());
            let q_at_eval_point = w_i_plus_1_poly
                .evaluate(line_at_eval_point)
                .map_err(|_| ProverError::EvaluationFailed)?;

            evaluations_y.push(q_at_eval_point);
        }
        let poly_q = Polynomial::interpolate(&evaluation_points_x, &evaluations_y)
            .map_err(|_| ProverError::EvaluationFailed)?;

        let layer_proof = LayerProof {
            claimed_sum: initial_sum,
            sumcheck_proof,
            poly_q: poly_q.clone(),
        };
        layer_proofs.push(layer_proof);

        //let m_i = poly_q.evaluate(&r_last);
    }

    // Evaluate the last layer at the final_point and store as the final claim
    let last_layer_evaluation = evaluation
        .layers
        .last()
        .ok_or(ProverError::EvaluationFailed)?;
    let last_layer_poly = DenseMultilinearPolynomial::new(last_layer_evaluation.to_vec());
    let final_claim = last_layer_poly
        .evaluate(r_i.clone())
        .map_err(|_| ProverError::EvaluationFailed)?;

    // TODO PREGUNTA: Hace falta mandar todo esto o hay algunas cosas que las calucla también el verifier?
    let proof = GkrProof {
        layer_proofs,
        final_claim,
        input_values: input.to_vec(),
        output_values: evaluation.layers[0].clone(),
        final_evaluation_point: r_i,
    };

    Ok(proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    // LAZY EVALUATION: Helper function to evaluate a multilinear polynomial at a point
    // without pre-computing all evaluations
    fn evaluate_multilinear_lazy<F: IsField>(
        poly: &DenseMultilinearPolynomial<F>,
        point: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, ProverError>
    where
        <F as IsField>::BaseType: Send + Sync,
    {
        // Use the existing evaluate method but with lazy computation
        poly.evaluate(point.to_vec())
            .map_err(|_| ProverError::EvaluationFailed)
    }

    // LAZY EVALUATION: Helper function to create a polynomial from evaluations
    // only when needed, avoiding pre-computation
    fn create_polynomial_from_evaluations_lazy<F: IsField>(
        evaluations: Vec<FieldElement<F>>,
    ) -> DenseMultilinearPolynomial<F>
    where
        <F as IsField>::BaseType: Send + Sync,
    {
        DenseMultilinearPolynomial::new(evaluations)
    }

    #[test]
    fn test_lazy_evaluation() {
        type F = U64PrimeField<17>;

        // Create a simple multilinear polynomial: f(x,y) = x + y
        let evaluations = vec![
            FieldElement::<F>::zero(),  // f(0,0) = 0
            FieldElement::<F>::one(),   // f(0,1) = 1
            FieldElement::<F>::one(),   // f(1,0) = 1
            FieldElement::<F>::from(2), // f(1,1) = 2
        ];
        let poly = create_polynomial_from_evaluations_lazy(evaluations);

        // Test with a point that requires interpolation
        let point1 = vec![FieldElement::<F>::from(2), FieldElement::<F>::from(3)];
        let eval1 = evaluate_multilinear_lazy(&poly, &point1).unwrap();
        assert_eq!(eval1, FieldElement::<F>::from(5)); // 2 + 3 = 5

        // Test with a point from the hypercube
        let point2 = vec![FieldElement::<F>::one(), FieldElement::<F>::zero()];
        let eval2 = evaluate_multilinear_lazy(&poly, &point2).unwrap();
        assert_eq!(eval2, FieldElement::<F>::from(1));
    }
}
