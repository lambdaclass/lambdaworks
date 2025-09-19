use std::cmp;

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

/// A $2k_{i+1}$ variate polynomial used for each step of GKR protocol.
///
/// $$
/// f^{i}_{r_i}(b, c) \coloneqq
/// \widetilde{add}_i(r_i, b, c)(\tilde{W}\_{i+1}(b) +
/// \tilde{W}\_{i+1}(c)) +
/// \widetilde{mul}_i(r_i, b, c)(\tilde{W}\_{i+1}(b) \cdot
/// \tilde{W}\_{i+1}(c))
/// $$
#[derive(Clone)]
pub struct W<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    add_i: DenseMultilinearPolynomial<F>,
    mul_i: DenseMultilinearPolynomial<F>,
    w_b: DenseMultilinearPolynomial<F>,
    w_c: DenseMultilinearPolynomial<F>,
}

impl<F> W<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    /// Create a new `W` polynomial.
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

    // fn evaluate(&self, point: &[F]) -> Option<F> {
    //     let (b, c) = point.split_at({
    //         let this = &self.w_b;
    //         this.num_vars
    //     });
    //     let add_e = self.add_i.evaluate(&point.into());
    //     let mul_e = self.mul_i.evaluate(&point.into());

    //     let w_b = self.w_b.evaluate(&b.into());
    //     let w_c = self.w_c.evaluate(&c.into());

    //     Some(add_e * (w_b + w_c) + mul_e * (w_b * w_c))
    // }

    fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self {
        let b_partial = partial_point
            .get(..cmp::min(self.w_b.num_vars(), partial_point.len()))
            .unwrap_or(&[]);
        let c_partial = partial_point.get(self.w_b.num_vars()..).unwrap_or(&[]);

        let add_i = self.add_i.fix_variables(partial_point);
        let mul_i = self.mul_i.fix_variables(partial_point);
        let w_b = self.w_b.fix_variables(b_partial);
        let w_c = self.w_c.fix_variables(c_partial);

        Self {
            add_i,
            mul_i,
            w_b,
            w_c,
        }
    }

    fn to_univariate(&self) -> Polynomial<FieldElement<F>> {
        let domain = [
            FieldElement::from(0),
            FieldElement::from(1),
            FieldElement::from(2),
        ];

        let evals: Vec<FieldElement<F>> = domain
            .iter()
            .map(|e| {
                self.fix_variables(&[e.clone()])
                    .to_evaluations()
                    .into_iter()
                    .sum()
            })
            .collect();

        let poly_g_j = Polynomial::interpolate(&domain, &evals).unwrap();

        poly_g_j
    }

    fn num_vars(&self) -> usize {
        self.add_i.num_vars()
    }

    fn to_evaluations(&self) -> Vec<FieldElement<F>> {
        // combine the evaluations of separate multilinear
        // extensions into a vector of evaluations of the
        // whole polynomial
        let w_b_evals = self.w_b.to_evaluations();
        let w_c_evals = self.w_c.to_evaluations();
        let add_i_evals = self.add_i.to_evaluations();
        let mul_i_evals = self.mul_i.to_evaluations();

        let mut res = vec![];
        for (b_idx, w_b_item) in w_b_evals.iter().enumerate() {
            for (c_idx, w_c_item) in w_c_evals.iter().enumerate() {
                let bc_idx = idx(c_idx, b_idx, self.w_b.num_vars());

                res.push(
                    add_i_evals[bc_idx].clone() * (w_b_item.clone() + w_c_item)
                        + mul_i_evals[bc_idx].clone() * (w_b_item.clone() * w_c_item),
                );
            }
        }

        res
    }
}

/// Combine indices of two variables into one to be able
/// to index into evaluations of polynomial.
fn idx(i: usize, j: usize, num_vars: usize) -> usize {
    (i << num_vars) | j
}

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

#[derive(Clone)]
pub struct GKRPoly<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    add_i_ext: DenseMultilinearPolynomial<F>,
    mul_i_ext: DenseMultilinearPolynomial<F>,
    w_next_ext: DenseMultilinearPolynomial<F>,
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
        w_next_ext: DenseMultilinearPolynomial<F>,
    ) -> Self {
        Self {
            add_i_ext: add_i_ext.clone(),
            mul_i_ext,
            w_next_ext,
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
        let w_next_evals = self.w_next_ext.to_evaluations();

        let num_vars = self.num_vars - 1;
        let mut result = Vec::with_capacity(1 << num_vars);

        println!("num vars: {:?}", num_vars);
        println!("add i evals: {:?}", add_i_evals.len());
        println!("mul i evals: {:?}", mul_i_evals.len());
        println!("w next number of evals: {:?}", w_next_evals.len());

        // Construct the GKR polynomial evaluations directly
        for c_idx in 0..(1 << num_vars) {
            println!("c idx: {:?}", c_idx);
            // 2^{k_{i+1}}. (00, ..., 11) = (0, ..., 3). 00
            for b_idx in 0..(1 << num_vars) {
                println!("b idx: {:?}", b_idx);

                // 01
                let bc_idx = b_idx + (c_idx << num_vars); // 0001
                println!("bc_idx: {:?}", bc_idx);
                let w_b = &w_next_evals[b_idx];
                let w_c = &w_next_evals[c_idx];
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
        let domain_points = [
            FieldElement::from(0),
            FieldElement::from(1),
            FieldElement::from(3),
        ];

        let eval_values: Vec<FieldElement<F>> = domain_points
            .iter()
            .map(|domain_point| {
                self.fix_variables(&[domain_point.clone()])
                    .get_hypercube_evaluations()
                    .into_iter()
                    .sum()
            })
            .collect();

        let poly_g_j = Polynomial::interpolate(&domain_points, &eval_values).unwrap();
        poly_g_j
    }

    fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self {
        let bc_partial_point = partial_point
            .get(..cmp::min(self.w_next_ext.num_vars(), partial_point.len()))
            .unwrap();

        let add_i_ext = self.add_i_ext.fix_variables(partial_point);
        let mul_i_ext = self.mul_i_ext.fix_variables(partial_point);
        let w_next_ext = self.w_next_ext.fix_variables(bc_partial_point);

        Self {
            add_i_ext: add_i_ext.clone(),
            mul_i_ext,
            w_next_ext,
            num_vars: add_i_ext.num_vars(),
        }
    }
}

#[derive(Clone)]
pub struct GKRSumcheck<F>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync,
{
    g: W<F>,
    challenges: Vec<FieldElement<F>>,
}

impl<F> GKRSumcheck<F>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    // // Returns proof of the form: (claimed_sum, proof_polys).
    // fn prove(
    //     mut self,
    //     transcript: &mut DefaultTranscript<F>,
    // ) -> (FieldElement<F>, Vec<Polynomial<FieldElement<F>>>) {
    //     let poly = self.poly;
    //     let num_vars = poly.num_vars;
    //     let claimed_sum = poly.clone().get_hypercube_evaluations().into_iter().sum();

    //     transcript.append_bytes(&num_vars.to_be_bytes());
    //     transcript.append_field_element(&claimed_sum);

    //     let mut proof_polys = Vec::with_capacity(num_vars);
    //     let mut challenges: Vec<FieldElement<F>> = Vec::with_capacity(num_vars);
    //     let mut current_challenge: Option<&FieldElement<F>> = None;

    //     // Execute rounds. One round for each variable.
    //     for j in 0..num_vars {
    //         // Prover computes the round polynomial g_j, fixing the first j variables and summing over the other ones.
    //         let g_j = poly.compute_univariate_poly(current_challenge);

    //         // Append g_j information to transcript for the verifier to derive challenge
    //         transcript.append_bytes(&j.to_be_bytes());
    //         let coeffs = g_j.coefficients();
    //         transcript.append_bytes(&coeffs.len().to_be_bytes());
    //         if coeffs.is_empty() {
    //             transcript.append_field_element(&FieldElement::zero());
    //         } else {
    //             for coeff in coeffs {
    //                 transcript.append_field_element(coeff);
    //             }
    //         }

    //         proof_polys.push(g_j);

    //         // Derive challenge for the next round from transcript (if not the last round)
    //         if j < num_vars - 1 {
    //             let challenge = transcript.sample_field_element();
    //             self.challenges.push(challenge.clone());
    //             current_challenge = self.challenges.last();
    //         } else {
    //             // No challenge needed after the last round polynomial is sent
    //             current_challenge = None;
    //         }
    //     }

    //     (claimed_sum.clone(), proof_polys)
    // }

    // Returns proof of the form: (claimed_sum, proof_polys).
    fn prove_2(
        mut self,
        transcript: &mut DefaultTranscript<F>,
    ) -> (FieldElement<F>, Vec<Polynomial<FieldElement<F>>>) {
        let g = self.g.clone();
        let num_vars = g.num_vars();
        let claimed_sum = g.clone().to_evaluations().into_iter().sum();

        transcript.append_bytes(&num_vars.to_be_bytes());
        transcript.append_field_element(&claimed_sum);

        let mut proof_polys = Vec::with_capacity(num_vars);
        // let mut r_j = FieldElement::<F>::one();
        let mut challenges: Vec<FieldElement<F>> = Vec::with_capacity(num_vars);
        let mut current_challenge: FieldElement<F> = FieldElement::<F>::one();

        // Execute rounds. One round for each variable.
        for j in 0..num_vars {
            if j != 0 {
                challenges.push(current_challenge.clone());
                self.g = self.g.clone().fix_variables(&[current_challenge.clone()]);
            }
            let g_j = self.g.clone().to_univariate();

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
                let challenge = transcript.sample_field_element();
                self.challenges.push(challenge.clone());
                current_challenge = challenge;
            }
            // } else {
            //     // No challenge needed after the last round polynomial is sent
            //     current_challenge = None;
            // }
        }

        (claimed_sum.clone(), proof_polys)
    }
}

/// A type of a gate in the Circuit.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GateType {
    /// An addition gate.
    Add,

    /// A multiplication gate.
    Mul,
}

/// A gate in the Circuit.
#[derive(Clone, Copy)]
pub struct Gate {
    /// A type of the gate.
    pub ttype: GateType,

    /// Two inputs, indexes into the previous layer gates outputs.
    pub inputs: [usize; 2],
}

impl Gate {
    /// Create a new `Gate`.
    pub fn new(ttype: GateType, inputs: [usize; 2]) -> Self {
        Self { ttype, inputs }
    }
}

/// A layer of gates in the circuit.
#[derive(Clone)]
pub struct CircuitLayer {
    pub layer: Vec<Gate>,
}

impl CircuitLayer {
    /// Create a new `CircuitLayer`.
    pub fn new(layer: Vec<Gate>) -> Self {
        Self { layer }
    }

    /// The length of the layer.
    pub fn len(&self) -> usize {
        self.layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layer.is_empty()
    }
}

/// An evaluation of a `Circuit` on some input.
/// Stores every circuit layer interediary evaluations and the
/// circuit evaluation outputs.
pub struct CircuitEvaluation<F> {
    /// Evaluations on per-layer basis.
    pub layers: Vec<Vec<F>>,
}

impl<F: Copy> CircuitEvaluation<F> {
    /// Takes a gate label and outputs the corresponding gate's value at layer `layer`.
    pub fn w(&self, layer: usize, label: usize) -> F {
        self.layers[layer][label]
    }
}

/// The circuit in layered form.
#[derive(Clone)]
pub struct Circuit {
    /// First layer being the output layer, last layer being
    /// the input layer.
    layers: Vec<CircuitLayer>,

    /// Number of inputs
    num_inputs: usize,
}

impl Circuit {
    pub fn new(layers: Vec<CircuitLayer>, num_inputs: usize) -> Self {
        Self { layers, num_inputs }
    }

    pub fn num_vars_at(&self, layer: usize) -> Option<usize> {
        let num_gates = if let Some(layer) = self.layers.get(layer) {
            layer.len()
        } else if layer == self.layers.len() {
            self.num_inputs
        } else {
            return None;
        };

        Some((num_gates as u64).trailing_zeros() as usize)
    }

    /// Evaluate a `Circuit` on a given input.
    pub fn evaluate<F>(&self, input: &[FieldElement<F>]) -> CircuitEvaluation<FieldElement<F>>
    where
        F: IsField,
    {
        let mut layers = vec![];
        let mut current_input = input.to_vec();

        layers.push(current_input.clone());

        for layer in self.layers.iter().rev() {
            let temp_layer: Vec<_> = layer
                .layer
                .iter()
                .map(|e| match e.ttype {
                    GateType::Add => {
                        current_input[e.inputs[0]].clone() + current_input[e.inputs[1]].clone()
                    }
                    GateType::Mul => {
                        current_input[e.inputs[0]].clone() * current_input[e.inputs[1]].clone()
                    }
                })
                .collect();

            layers.push(temp_layer.clone());
            current_input = temp_layer;
        }

        layers.reverse();
        CircuitEvaluation { layers }
    }

    /// The $\text{add}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn add_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];

        gate.ttype == GateType::Add && gate.inputs[0] == b && gate.inputs[1] == c
    }

    /// The $\text{mul}_i(a, b, c)$ polynomial value at layer $i$.
    pub fn mul_i(&self, i: usize, a: usize, b: usize, c: usize) -> bool {
        let gate = &self.layers[i].layer[a];

        gate.ttype == GateType::Mul && gate.inputs[0] == b && gate.inputs[1] == c
    }

    pub fn layers(&self) -> &[CircuitLayer] {
        &self.layers
    }

    pub fn num_outputs(&self) -> usize {
        self.layers[0].layer.len()
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn add_i_ext<F: IsField>(
        &self,
        r_i: &[FieldElement<F>],
        i: usize,
    ) -> DenseMultilinearPolynomial<F>
    where
        F::BaseType: Send + Sync + Copy,
    {
        let mut add_i_evals: Vec<FieldElement<F>> = vec![];
        // CHANGE THIS. put it in the struct
        let num_vars_current = (self.layers[i].len() as f64).log2() as usize;

        let num_vars_next = (self
            .layers
            .get(i + 1)
            .map(|c| c.len())
            .unwrap_or(self.num_inputs) as f64)
            .log2() as usize;

        // TODO: CHANGE THIS FUNCTION.
        // Make a vector of length num_vars_current + 2 * num_vars_next full of zeros.
        // Después recorrer los gates del layer i, y para cada gate ahí vemos qué tipo de layer es y en qué posición está. Para la posición que está metemos un 1.
        for a in 0..1 << num_vars_current {
            for b in 0..1 << num_vars_next {
                for c in 0..1 << num_vars_next {
                    add_i_evals.push(if self.add_i(i, a, b, c) {
                        FieldElement::one()
                    } else {
                        FieldElement::zero()
                    });
                }
            }
        }

        let add_i = DenseMultilinearPolynomial::new(add_i_evals);
        let mut p = add_i;
        for (_i, val) in r_i.iter().enumerate() {
            p = p.fix_first_variable(val);
        }
        p
    }

    pub fn mul_i_ext<F: IsField>(
        &self,
        r_i: &[FieldElement<F>],
        i: usize,
    ) -> DenseMultilinearPolynomial<F>
    where
        F::BaseType: Send + Sync + Copy,
    {
        let mut mul_i_evals: Vec<FieldElement<F>> = vec![];
        let num_vars_current = (self.layers[i].len() as f64).log2() as usize;

        let num_vars_next = (self
            .layers
            .get(i + 1)
            .map(|c| c.len())
            .unwrap_or(self.num_inputs) as f64)
            .log2() as usize;

        for a in 0..1 << num_vars_current {
            for b in 0..1 << num_vars_next {
                for c in 0..1 << num_vars_next {
                    mul_i_evals.push(if self.mul_i(i, a, b, c) {
                        FieldElement::one()
                    } else {
                        FieldElement::zero()
                    });
                }
            }
        }

        let mul_i = DenseMultilinearPolynomial::new(mul_i_evals);
        let mut p = mul_i;
        for (_i, val) in r_i.iter().enumerate() {
            p = p.fix_first_variable(val);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    use super::*;
    const MODULUS23: u64 = 23;
    type F23 = U64PrimeField<MODULUS23>;
    type F23E = FieldElement<F23>;

    fn circuit_from_post() -> Circuit {
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

    // #[test]
    // fn sumchek_prints() {
    //     let circuit = circuit_from_post();
    //     let input = [F23E::from(3), F23E::from(1)];
    //     let circuit_evaluation = circuit.evaluate(&input);

    //     println!("layer 1: {:?}", circuit_evaluation.layers[1]);

    //     let w_next_ext = DenseMultilinearPolynomial::new(circuit_evaluation.layers[1].clone());

    //     let poly = GKRPoly::<F23>::new(
    //         circuit.add_i_ext(&[], 0),
    //         circuit.mul_i_ext(&[], 0),
    //         w_next_ext,
    //     );

    //     let transcript = &mut DefaultTranscript::<F23>::default();
    //     let challenges = Vec::new();
    //     let sumcheck_test = GKRSumcheck::<F23> { poly, challenges };

    //     let proof = sumcheck_test.prove(transcript);
    //     println!("sumchek proof: {:?}", proof);
    // }

    #[test]
    fn thaler_sumcheck_prints() {
        let circuit = circuit_from_post();
        let input = [F23E::from(3), F23E::from(1)];
        let circuit_evaluation = circuit.evaluate(&input);

        let add_i = circuit.add_i_ext(&[F23E::from(2)], 0);
        let mul_i = circuit.mul_i_ext(&[F23E::from(2)], 0);

        let w_b = DenseMultilinearPolynomial::new(circuit_evaluation.layers[1].clone());
        let w_c = DenseMultilinearPolynomial::new(circuit_evaluation.layers[1].clone());

        let poly = W::<F23>::new(add_i, mul_i, w_b, w_c);

        let hypercube_evaluations = poly.to_evaluations();
        println!("hypercube evaluations: {:?}", hypercube_evaluations);
        println!(
            "hypercube evaluations len: {:?}",
            hypercube_evaluations.len()
        );

        let g0 = poly.to_univariate();

        println!("g0: {:?}", g0);

        let g = poly;
        let challenges = Vec::new();

        let sumcheck_test = GKRSumcheck::<F23> { g, challenges };
        let transcript = &mut DefaultTranscript::<F23>::default();
        let proof = sumcheck_test.prove_2(transcript);
        println!("proof: {:?}", proof);
    }

    #[test]
    fn enteros() {
        // let a = F23E::from(16) * F23E::from(6).inv().unwrap();
        // let b = F23E::from(21) * F23E::from(11).inv().unwrap();
        // println!("a: {:?}  y b: {:?}", a, b);
        let g_at_2 = F23E::from(7) * F23E::from(7) + F23E::from(6) * F23E::from(7).square();
        let g_post_at_2 = F23E::from(2) * F23E::from(7) + F23E::from(11) * F23E::from(7).square();
        print!("g at 2: {:?} y g post at 2: {:?}", g_at_2, g_post_at_2);
    }
}
