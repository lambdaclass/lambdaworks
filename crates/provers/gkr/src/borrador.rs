use lambdaworks_crypto::fiat_shamir::{
    default_transcript::DefaultTranscript, is_transcript::IsTranscript,
};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    traits::ByteConversion,
};
use thiserror::Error;

// #[derive(Error, Debug)]
// pub enum SumcheckError<F: IsField> {
//     #[error("prover claim mismatches evaluation {0} {1}")]
//     ProverClaimMismatch(String, String),
//     #[error("verifier has no oracle access to the polynomial")]
//     NoPolySet,
//     #[error("evaluation failed")]
//     EvaluationFailed,
//     #[error("interpolation failed")]
//     InterpolationFailed,
//     #[error("invalid proof")]
//     InvalidProof,
// }

pub struct GKRPoly<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    add_i_ext: DenseMultilinearPolynomial<F>,
    mul_i_ext: DenseMultilinearPolynomial<F>,
    w_b_ext: DenseMultilinearPolynomial<F>,
    w_c_ext: DenseMultilinearPolynomial<F>,
    num_vars: usize,
}

impl<F> GKRPoly<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(
        add_i_ext: DenseMultilinearPolynomial<F>,
        mul_i_ext: DenseMultilinearPolynomial<F>,
        w_b_ext: DenseMultilinearPolynomial<F>,
        w_c_ext: DenseMultilinearPolynomial<F>,
    ) -> Self {
        Self {
            add_i_ext: add_i_ext.clone(),
            mul_i_ext,
            w_b_ext,
            w_c_ext,
            num_vars: add_i_ext.num_vars(),
        }
    }

    // pub fn evaluate(&self, point: &[FieldElement<F>]) -> Result<FieldElement<F>, SumcheckError<F>> {
    //     if point.len() != self.num_vars {
    //         return Err(SumcheckError::EvaluationFailed);
    //     }

    //     let add_i_eval = self
    //         .add_i_ext
    //         .evaluate(point.to_vec())
    //         .map_err(|_| SumcheckError::EvaluationFailed)?;
    //     let mul_i_eval = self
    //         .mul_i_ext
    //         .evaluate(point.to_vec())
    //         .map_err(|_| SumcheckError::EvaluationFailed)?;
    //     let w_b_eval = self
    //         .w_b_ext
    //         .evaluate(point.to_vec())
    //         .map_err(|_| SumcheckError::EvaluationFailed)?;
    //     let w_c_eval = self
    //         .w_c_ext
    //         .evaluate(point.to_vec())
    //         .map_err(|_| SumcheckError::EvaluationFailed)?;

    //     // GKR polynomial: add_i * (w_b + w_c) + mul_i * w_b * w_c
    //     let w_sum = w_b_eval + w_c_eval;
    //     let term1 = add_i_eval * w_sum;
    //     let term2 = mul_i_eval * w_b_eval * w_c_eval;

    //     Ok(term1 + term2)
    // }

    pub fn get_hypercube_evaluations(self) -> Vec<FieldElement<F>> {
        let add_i_evals = self.add_i_ext.to_evaluations();
        let mul_i_evals = self.mul_i_ext.to_evaluations();
        let w_b_evals = self.w_b_ext.to_evaluations();
        let w_c_evals = self.w_c_ext.to_evaluations();
        let num_vars = self.num_vars;
        let mut result = Vec::with_capacity(1 << num_vars);

        // Construct the GKR polynomial evaluations directly
        for c_idx in 0..(1 << num_vars) {
            // 2^{k_{i+1}}. (00, ..., 11) = (0, ..., 3). 00
            for b_idx in 0..(1 << num_vars) {
                // 01
                let bc_idx = c_idx + (b_idx << num_vars); // 0001
                let w_b = &w_b_evals[b_idx];
                let w_c = &w_c_evals[c_idx];
                let gkr_eval =
                    &add_i_evals[bc_idx] * (w_b + w_c) + &mul_i_evals[bc_idx] * (w_b * w_c);
                result.push(gkr_eval);
            }
        }

        result
    }

    fn compute_univariate_poly(
        &self,
        challenge_prev: Option<&FieldElement<F>>,
    ) -> Polynomial<FieldElement<F>> {
        let mut eval_domain_x = Vec::with_capacity(3);
        let mut eval_values_y = Vec::with_capacity(3);

        let domain_points = [
            FieldElement::from(0),
            FieldElement::from(1),
            FieldElement::from(3),
        ];

        let eval_values = domain_points
            .iter()
            .map(|domain_point| {
                self.fix_variable(&domain_point)
                    .get_hypercube_evaluations()
                    .into_iter()
                    .sum()
            })
            .collect();

        let poly_g_j = Polynomial::interpolate(&domain_points, &eval_values).unwrap();
        poly_g_j
    }

    fn fix_variable(&self, prefix_point: &FieldElement<F>) -> Self {
        let add_i = self.add_i_ext.fix_last_variable(prefix_point);
        let mul_i = self.mul_i_ext.fix_last_variable(prefix_point);
        let w_b = self.w_b_ext.fix_last_variable(b_partial);
        let w_c = self.w_c_ext.fix_last_variable(c_partial);

        Self {
            add_i,
            mul_i,
            w_b,
            w_c,
        }
    }
}

pub struct GKRSumcheck<F>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync,
{
    poly: GKRPoly<F>,
    transcript: DefaultTranscript<F>,
    challenges: Vec<FieldElement<F>>,
}

impl<F> GKRSumcheck<F>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    // Returns proof of the form: (claimed_sum, proof_polys).
    fn prove(self) -> (FieldElement<F>, Vec<Polynomial<FieldElement<F>>>) {
        let poly = self.poly;
        let num_vars = poly.num_vars;
        let claimed_sum = self.poly.get_hypercube_evaluations().into_iter().sum();

        let mut transcript = self.transcript;

        transcript.append_bytes(&num_vars.to_be_bytes());
        transcript.append_field_element(&claimed_sum);

        let mut proof_polys = Vec::with_capacity(num_vars);
        let mut challenges = Vec::with_capacity(num_vars);
        let mut current_challenge: Option<FieldElement<F>> = None;

        // Execute rounds. One round for each variable.
        for j in 0..num_vars {
            // Prover computes the round polynomial g_j, fixing the first j variables and summing over the other ones.
            let g_j = poly.compute_univariate_poly(&poly, current_challenge);

            // Append g_j information to transcript for the verifier to derive challenge
            transcript.append_bytes(&j.to_be_bytes());
            let coeffs = g_j.coefficients();
            transcript.append_bytes(&coeffs.len().to_be_bytes());
            if coeffs.is_empty() {
                transcript.append_field_element(&FieldElement::zero());
            } else {
                for coeff in coeffs {
                    transcript.append_field_element(coeff);
                }
            }

            proof_polys.push(g_j);

            // Derive challenge for the next round from transcript (if not the last round)
            if j < num_vars - 1 {
                let mut challenge = transcript.sample_field_element();
                self.challenges.push(challenge.clone());
                current_challenge = Some(challenge);
            } else {
                // No challenge needed after the last round polynomial is sent
                current_challenge = None;
            }
        }

        (claimed_sum, proof_polys)
    }
}

//     pub fn verify(&self, proof)-> Result<bool,SumcheckError>
//     where
//     F: IsField + HasDefaultTranscript,
//     FieldElement<F>: ByteConversion,

// {
//     self.transcript.append_felt(&proof.claimed_sum);

//     let mut current_sum = proof.claimed_sum.clone();
//     let mut challenges = Vec::new();

//     for (round, g_j) in proof.round_polynomials.iter().enumerate() {

//         let round_label = format!("round_{}_poly", round);
//         self.transcript.append_bytes(round_label.as_bytes());

//         let coeffs = g_j.coefficients();
//         self.transcript
//             .append_bytes(&(coeffs.len() as u64).to_be_bytes());
//         if coeffs.is_empty() {
//             self.transcript.append_felt(&FieldElement::zero());
//         } else {
//             for coeff in coeffs {
//                 self.transcript.append_felt(coeff);
//             }
//         }

//         let g_j_0 = g_j.evaluate(&FieldElement::zero());
//         let g_j_1 = g_j.evaluate(&FieldElement::one());
//         let sum_evals = g_j_0 + g_j_1;

//         if sum_evals != current_sum {
//             return Err(SumcheckError::ProverClaimMismatch(
//                 format!("{:?}", sum_evals),
//                 format!("{:?}", current_sum),
//             ));
//         }

//         let r_j = self.transcript.sample_field_element();
//         challenges.push(r_j.clone());
//         current_sum = g_j.evaluate(&r_j);

//         if round == proof.round_polynomials.len() - 1 {
//             let final_eval = self.poly.evaluate(&challenges)?;
//             return Ok(final_eval == current_sum);
//         }
//     }

//     Err(SumcheckError::InvalidProof)
// }
//     }
