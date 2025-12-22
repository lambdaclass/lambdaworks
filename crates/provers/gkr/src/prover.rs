use crate::sumcheck::GKRSumcheckProof;
use crate::{circuit::Circuit, sumcheck::gkr_sumcheck_prove};

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::dense_multilinear_poly::DenseMultilinearPolynomial,
    traits::ByteConversion,
};

type GKRPolynomialTerms<F> = (
    Vec<DenseMultilinearPolynomial<F>>,
    Vec<DenseMultilinearPolynomial<F>>,
);
#[derive(Debug)]
pub enum ProverError {
    MultilinearPolynomialEvaluationError,
    SumcheckError,
    CircuitError,
    InterpolationError,
}

/// Each layer proof contains the sumcheck round polynomials g_j, the sumcheck challenges and the polynomial q.
/// Recall that the polynomial q is the composition of the circuit evaluation polynomial extension W_{i+1} and the line function l.
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
    /// Constructs the polynomial `f(b, c)` that the GKR sumcheck will use for a specific layer and fixed values `r_i`.
    ///
    /// The polynomial `f` is defined as:
    /// `f(b, c) = add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c)) + mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))`
    ///
    /// where:
    /// - `add_i` and `mul_i` are the multilinear polynomial extensions for the addition and multiplication gates of layer `i`.
    /// - `r_i` contains the random challenges  for layer `i`.
    /// - `W_{i+1}` is the multilinear polynomial extension representing the circuit evaluations of layer `i+1`.
    /// - `b` and `c` are the inputs to the gates, with `k_{i+1}` variables each.
    ///
    /// See our blog post: <https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/>
    ///
    /// This function is only known to the prover.
    ///
    /// It returns the components of `f` grouped into two terms, ready to be used by the sumcheck prover, which expects
    /// a product of multilinear polynomials.
    /// Both terms are the product of two multilinear polynomials.
    /// The first term is  `add_i(r_i, b, c) * (W_{i+1}(b) + W_{i+1}(c))`.
    /// And the second term is `mul_i(r_i, b, c) * (W_{i+1}(b) * W_{i+1}(c))`.
    fn build_gkr_polynomial<F: IsField + Clone>(
        circuit: &Circuit,
        r_i: &[FieldElement<F>],
        w_next_evals: &[FieldElement<F>],
        layer_idx: usize,
    ) -> Result<GKRPolynomialTerms<F>, ProverError>
    where
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        // Get the multilinear extensions of the wiring predicates `add(a, b, c)` and
        // `mul(a, b, c)` with the variable `a` fixed at `r_i`.
        let add_i_ext = circuit.add_i_ext::<F>(r_i, layer_idx);
        let mul_i_ext = circuit.mul_i_ext::<F>(r_i, layer_idx);

        let num_vars_next = circuit
            .num_vars_at(layer_idx + 1)
            .ok_or(ProverError::CircuitError)?;

        let mut w_sum_evals: Vec<FieldElement<F>> = Vec::new();
        let mut w_mul_evals: Vec<FieldElement<F>> = Vec::new();
        let next_layer_size = 1 << num_vars_next;

        for c_idx in 0..next_layer_size {
            for b_idx in 0..next_layer_size {
                let w_b = &w_next_evals[b_idx];
                let w_c = &w_next_evals[c_idx];
                w_sum_evals.push(w_b + w_c);
                w_mul_evals.push(w_b * w_c);
            }
        }
        // w_sum_ext(b, c) = W_{i+1}(b) + W_{i+1}(c)
        let w_sum_ext = DenseMultilinearPolynomial::new(w_sum_evals);
        // w_mul_ext(b, c) = W_{i+1}(b) * W_{i+1}(c)
        let w_mul_ext = DenseMultilinearPolynomial::new(w_mul_evals);

        // Corresponds to add_i(r_i,b,c) * (W_{i+1}(b) + W_{i+1}(c))
        let term1 = vec![add_i_ext, w_sum_ext];
        // Corresponds to mul_i(r_i,b,c) * W_{i+1}(b) * W_{i+1}(c)
        let term2 = vec![mul_i_ext, w_mul_ext];

        Ok((term1, term2))
    }

    /// Given `b` and `c` (challenges for the input of the gates),
    /// and given the evaluations of W_{i+1} (the gates evaluations),
    /// it returns the composition polynomial `q = W_{i+1} o l`, where `l` is the line function from `b` to `c`.
    /// It builds the polynomial `q` by interpolating three points (0, 1 and 2).
    /// There is no need to interpolate more points because `q` has degree 2 (since `l` is linear and
    /// `W` is a 2-variable multlinear polynomial).
    fn build_polynomial_q<F>(
        b: &[FieldElement<F>],
        c: &[FieldElement<F>],
        w_next_evals: Vec<FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError>
    where
        F: IsField,
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let mut domain_points_x = Vec::new();
        let mut evaluation_values_y = Vec::new();
        let w_next_poly = DenseMultilinearPolynomial::new(w_next_evals);
        for i in 0..3 {
            let eval_point_x = FieldElement::from(i as u64);
            let line_at_eval_point = crate::line(b, c, &eval_point_x);
            let q_at_eval_point = w_next_poly
                .evaluate(line_at_eval_point)
                .map_err(|_| ProverError::MultilinearPolynomialEvaluationError)?;

            domain_points_x.push(eval_point_x);
            evaluation_values_y.push(q_at_eval_point);
        }
        let poly_q = Polynomial::interpolate(&domain_points_x, &evaluation_values_y)
            .map_err(|_| ProverError::InterpolationError)?;
        Ok(poly_q)
    }

    /// Generate a GKR proof.
    ///
    /// The GKR protocol reduces a claim about layer `i` to a claim about layer `i+1`, from the top layer (outputs) to the bottom layer (inputs).
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
        if input.len() != circuit.num_inputs() {
            return Err(ProverError::CircuitError);
        }

        // The prover evaluates the circuit.
        let evaluation = circuit.evaluate(input);

        // Fiat-Shamir heuristic:
        // Both parties need to to append to the transcript the circuit, the inputs and the outputs.
        // See https://eprint.iacr.org/2025/118.pdf, Sections 2.1 and 2.2.
        let mut transcript = DefaultTranscript::<F>::default();
        // Append the circuit data to the transcript.
        transcript.append_bytes(&crate::circuit_to_bytes(circuit));
        // Append public inputs x.
        for x in input {
            transcript.append_bytes(&x.to_bytes_be());
        }
        // Append public outputs y (first layer of evaluation).
        for y in &evaluation.layers[0] {
            transcript.append_bytes(&y.to_bytes_be());
        }

        let k_0 = circuit.num_vars_at(0).ok_or(ProverError::CircuitError)?;
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        let mut layer_proofs = Vec::new();

        // For each layer, the prover makes a layer proof applying the sumcheck protocol to the function `f`.
        for layer_idx in 0..circuit.layers().len() {
            // Evaluations of W_{i+1} multilinear polynomial extension.
            let w_next_evals = &evaluation.layers[layer_idx + 1];

            // Build the GKR polynomial terms
            let (term_1, term_2) =
                Prover::build_gkr_polynomial(circuit, &r_i, w_next_evals, layer_idx)?;

            let sumcheck_proof = gkr_sumcheck_prove(term_1, term_2, &mut transcript)?;

            let sumcheck_challenges = &sumcheck_proof.challenges;

            // After applying the sumcheck, the prover needs to sample the challenges for the next layer and
            // build the polynomial q = W o l.
            let num_vars_next = circuit
                .num_vars_at(layer_idx + 1)
                .ok_or(ProverError::CircuitError)?;

            // r* in our blog post <https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/>
            let r_new = transcript.sample_field_element();

            // Construct the next round's random point using line function
            //  l(x) = b + x * (c - b)
            let (b, c) = sumcheck_challenges.split_at(num_vars_next);
            // r_i = l(r_new)
            r_i = crate::line(b, c, &r_new);

            let poly_q = Prover::build_polynomial_q(b, c, w_next_evals.clone())?;

            let layer_proof = LayerProof {
                sumcheck_proof,
                poly_q,
            };
            layer_proofs.push(layer_proof);
        }

        let output_values = evaluation.layers[0].clone();
        let proof = GKRProof {
            input_values: input.to_vec(),
            output_values,
            layer_proofs,
        };

        Ok(proof)
    }
}
