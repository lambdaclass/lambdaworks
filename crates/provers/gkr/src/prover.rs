use crate::sumcheck::GKRSumcheckProof;
use crate::{circuit::Circuit, sumcheck::gkr_sumcheck_prove};

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
    MultilinearPolynomialEvaluationError,
    SumcheckError,
    CircuitError,
    InterpolationError,
}

#[derive(Debug, Clone)]
pub struct LayerProof<F: IsField> {
    pub sumcheck_proof: GKRSumcheckProof<F>,
    pub poly_q: Polynomial<FieldElement<F>>,
}

/// A GKR proof.
#[derive(Debug, Clone)]
pub struct GKRProof<F: IsField> {
    pub input_values: Vec<FieldElement<F>>,
    pub output_values: Vec<FieldElement<F>>,
    pub layer_proofs: Vec<LayerProof<F>>,
}

pub struct Prover;

impl Prover {
    /// Constructs the polynomial `f(b, c)` for the GKR sumcheck for a specific layer and fixed values `r_i`.
    ///
    /// The polynomial `f` is defined as:
    /// `f`(b, c) = add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c)) + mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))`
    ///
    /// where:
    /// - `add_i` and `mul_i` are the multilinear polynomial extensions for the addition and multiplication gates of layer `i`.
    /// - `r_i` contains the random challenges  for layer `i`.
    /// - `W_{i+1}` is the multilinear polynomial extension representing the circuit evaluations of layer `i+1`.
    /// - `b` and `c` are the inputs to the gates, with `k_{i+1}` variables each.
    ///
    /// This function is only known to the prover. It returns the components of `f` grouped
    /// into two terms, ready to be used by the sumcheck prover, which expects a product of multilinear polynomials.
    /// Both terms are the product of two multilinear polynomials.
    /// The first term is  `add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c))`.
    /// And the second term is `mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))`.
    fn build_gkr_polynomial<F: IsField + Clone>(
        circuit: &Circuit,
        r_i: &[FieldElement<F>],
        w_next_evals: &[FieldElement<F>],
        layer_idx: usize,
    ) -> Result<Vec<Vec<DenseMultilinearPolynomial<F>>>, ProverError>
    where
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        // Get the multilinear extensions of the wiring predicates `add(a, b, c)` and
        // `mul(a, b, c)`` with the variable `a` fixed at `r_i`.
        let add_i_ext = circuit.add_i_ext::<F>(r_i, layer_idx);
        let mul_i_ext = circuit.mul_i_ext::<F>(r_i, layer_idx);

        let num_vars_next = circuit
            .num_vars_at(layer_idx + 1)
            .ok_or(ProverError::CircuitError)?;

        let mut w_sum_evals: Vec<FieldElement<F>> = Vec::new();
        let mut w_mul_evals: Vec<FieldElement<F>> = Vec::new();

        for c_idx in 0..(1 << num_vars_next) {
            for b_idx in 0..(1 << num_vars_next) {
                let w_b = &w_next_evals[b_idx];
                let w_c = &w_next_evals[c_idx];
                w_sum_evals.push(w_b + w_c);
                w_mul_evals.push(w_b * w_c);
            }
        }

        let w_sum_ext = DenseMultilinearPolynomial::new(w_sum_evals);
        let w_mul_ext = DenseMultilinearPolynomial::new(w_mul_evals);

        // Corresponds to add_i(r_i,b,c) * (W_{i+1}(b) + W_{i+1}(c))
        let term1 = vec![add_i_ext, w_sum_ext];
        // Corresponds to mul_i(r_i,b,c) * W_{i+1}(b) * W_{i+1}(c)
        let term2 = vec![mul_i_ext, w_mul_ext];

        Ok(vec![term1, term2])
    }

    /// Generate a GKR proof.
    ///
    /// The GKR protocol reduces a claim about layer `i` to a claim about layer `i+1`, from the top (outputs) to the bottom (inputs).
    /// This reduction is proven using a sumcheck protocol on the polynomial `f` that is
    /// constructed by the prover using the function `build_gkr_polynomial`.
    pub fn generate_proof<F>(
        circuit: &Circuit,
        input: &[FieldElement<F>],
    ) -> Result<GKRProof<F>, ProverError>
    where
        F: IsField + HasDefaultTranscript,
        FieldElement<F>: ByteConversion,
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let evaluation = circuit.evaluate(input);

        let mut layer_proofs = Vec::new();

        // Fiat-Shamir heuristic:
        // Both parties need to to append to the transcript the circuit, the inputs and the outputs.
        // See https://eprint.iacr.org/2025/118.pdf, Sections 2.1 and 2.2
        let mut transcript = DefaultTranscript::<F>::default();
        // 1. Append the circuit data to the transcript.
        transcript.append_bytes(&crate::circuit_to_bytes(circuit));
        // 2. x public inputs
        for x in input {
            transcript.append_bytes(&x.to_bytes_be());
        }
        // 3. y outputs (first layer of evaluation)
        for y in &evaluation.layers[0] {
            transcript.append_bytes(&y.to_bytes_be());
        }

        let k_0 = circuit.num_vars_at(0).ok_or(ProverError::CircuitError)?;
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        for layer_idx in 0..circuit.layers().len() {
            let w_i_plus_1 = &evaluation.layers[layer_idx + 1];

            // Build the GKR polynomial terms
            let gkr_poly_terms =
                Prover::build_gkr_polynomial(circuit, &r_i, w_i_plus_1, layer_idx)?;

            let sumcheck_proof = gkr_sumcheck_prove(gkr_poly_terms.clone(), &mut transcript)?;

            let sumcheck_challenges = &sumcheck_proof.challenges;

            // Sample challenges for the next round using line function
            let k_i_plus_1 = circuit
                .num_vars_at(layer_idx + 1)
                .ok_or(ProverError::CircuitError)?;

            // r* in the Lambda post
            let r_last = transcript.sample_field_element();

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

                // q = W o l
                // q is the composition of W and the line
                let line_at_eval_point = crate::line(b, c, &eval_point_x);
                let w_i_plus_1_poly = DenseMultilinearPolynomial::new(w_i_plus_1.clone());
                let q_at_eval_point = w_i_plus_1_poly
                    .evaluate(line_at_eval_point)
                    .map_err(|_| ProverError::MultilinearPolynomialEvaluationError)?;

                evaluations_y.push(q_at_eval_point);
            }
            let poly_q = Polynomial::interpolate(&evaluation_points_x, &evaluations_y)
                .map_err(|_| ProverError::InterpolationError)?;

            let layer_proof = LayerProof {
                sumcheck_proof,
                poly_q: poly_q.clone(),
            };
            layer_proofs.push(layer_proof);
        }

        let proof = GKRProof {
            input_values: input.to_vec(),
            output_values: evaluation.layers[0].clone(),
            layer_proofs,
        };

        Ok(proof)
    }
}
