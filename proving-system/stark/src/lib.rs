pub mod air;
pub mod fri;

use air::constraints::evaluator::ConstraintEvaluator;
use air::constraints::helpers;
use air::frame::Frame;
use air::trace::TraceTable;
use air::AIR;
use fri::fri;
use lambdaworks_math::field::traits::IsTwoAdicField;
use lambdaworks_math::traits::ByteConversion;

use fri::fri_decommit::{fri_decommit_layers, FriDecommitment};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::polynomial::{self, Polynomial};

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::u256_two_adic_prime_field::U256MontgomeryTwoAdicPrimeField;
use lambdaworks_math::field::traits::IsField;

pub struct ProofConfig {
    pub count_queries: usize,
    pub blowup_factor: usize,
}

pub type PrimeField = U256MontgomeryTwoAdicPrimeField;
pub type FE = FieldElement<PrimeField>;

// DEFINITION OF CONSTANTS

const ORDER_OF_ROOTS_OF_UNITY_TRACE: u64 = 32;
const ORDER_OF_ROOTS_OF_UNITY_FOR_LDE: u64 = 1024;

// We are using 3 as the offset as it's our field's generator.
const COSET_OFFSET: u64 = 3;

// DEFINITION OF FUNCTIONS

/// This function takes a roots of unity and a coset factor
/// If coset_factor is 1, it's just expanding the roots of unity
/// w ^ 0, w ^ 1, w ^ 2 .... w ^ n-1
/// If coset_factor is h
/// h * w ^ 0, h * w ^ 1 .... h * w ^ n-1
pub fn generate_roots_of_unity_coset<F: IsField>(
    coset_factor: u64,
    primitive_root: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let coset_factor: FieldElement<F> = coset_factor.into();

    let mut numbers = vec![coset_factor.clone()];
    let mut exp: u64 = 1;
    let mut next_root = primitive_root.pow(exp) * &coset_factor;
    while next_root != coset_factor {
        numbers.push(next_root);
        exp += 1;
        next_root = primitive_root.pow(exp) * &coset_factor;
    }
    numbers
}

#[derive(Debug, Clone)]
pub struct StarkQueryProof<F: IsField> {
    pub trace_ood_frame_evaluations: Frame<F>,
    pub composition_poly_evaluations: Vec<FieldElement<F>>,
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub fri_decommitment: FriDecommitment<F>,
}

pub struct StarkProof<F: IsField> {
    pub trace_lde_poly_root: FieldElement<F>,
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub query_list: Vec<StarkQueryProof<F>>,
}

pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
pub use lambdaworks_crypto::merkle_tree::DefaultHasher;

pub fn fibonacci_trace<F: IsField>(initial_values: [FieldElement<F>; 2]) -> Vec<FieldElement<F>> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_values[0].clone());
    ret.push(initial_values[1].clone());

    for i in 2..(ORDER_OF_ROOTS_OF_UNITY_TRACE as usize) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    ret
}

pub fn prove<F: IsField + IsTwoAdicField, A: AIR + AIR<Field = F>>(
    trace: &[FieldElement<F>],
    air: &A,
) -> StarkQueryProof<F>
where
    FieldElement<F>: ByteConversion,
{
    let transcript = &mut Transcript::new();
    // let mut query_list = Vec::<StarkQueryProof>::new();

    // * Generate Coset
    let trace_primitive_root =
        F::get_primitive_root_of_unity(air.context().trace_length as u64).unwrap();

    let root_order = air.context().trace_length.trailing_zeros();
    let trace_roots_of_unity = F::get_powers_of_primitive_root_coset(
        root_order as u64,
        air.context().trace_length,
        &FieldElement::<F>::one(),
    )
    .unwrap();

    let lde_root_order =
        (air.context().trace_length * air.options().blowup_factor as usize).trailing_zeros();
    let lde_roots_of_unity_coset = F::get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        air.context().trace_length * air.options().blowup_factor as usize,
        &FieldElement::<F>::from(COSET_OFFSET),
    )
    .unwrap();

    let trace_poly = Polynomial::interpolate(&trace_roots_of_unity, trace);
    let lde_trace = trace_poly.evaluate_slice(&lde_roots_of_unity_coset);

    // TODO: Fiat-Shamir
    // z is the Out of domain evaluation point used in Deep FRI. It needs to be a point outside
    // of both the roots of unity and its corresponding coset used for the lde commitment.
    let z = FieldElement::from(3);
    let z_squared = &z * &z;

    let lde_trace = TraceTable::new(lde_trace, 1);

    // Create evaluation table
    let evaluator = ConstraintEvaluator::new(air, &trace_poly, &trace_primitive_root);

    // TODO: Fiat-Shamir
    let alpha = FieldElement::one();
    let beta = FieldElement::one();

    let alpha_and_beta_transition_coefficients = vec![(alpha.clone(), beta.clone())];
    let constraint_evaluations = evaluator.evaluate(
        &lde_trace,
        &lde_roots_of_unity_coset,
        &alpha_and_beta_transition_coefficients,
        (&alpha, &beta),
    );

    // Get composition poly
    let composition_poly =
        constraint_evaluations.compute_composition_poly(&lde_roots_of_unity_coset);

    let (composition_poly_even, composition_poly_odd) = composition_poly.even_odd_decomposition();
    // Evaluate H_1 and H_2 in z^2.
    let composition_poly_evaluations = vec![
        composition_poly_even.evaluate(&z_squared),
        composition_poly_odd.evaluate(&z_squared),
    ];

    let trace_ood_frame_evaluations = Frame::<F>::construct_ood_frame(
        &[trace_poly.clone()],
        &z,
        &air.context().transition_offsets,
        &trace_primitive_root,
    );

    // END EVALUATION BLOCK

    // Compute DEEP composition polynomial so we can commit to it using FRI.
    let mut deep_composition_poly = compute_deep_composition_poly(
        &trace_poly,
        &composition_poly_even,
        &composition_poly_odd,
        &z,
        &trace_primitive_root,
    );

    // * Do FRI on the composition polynomials
    let lde_fri_commitment = fri(
        &mut deep_composition_poly,
        &lde_roots_of_unity_coset,
        transcript,
    );

    // TODO: Fiat-Shamir
    let fri_decommitment_index: usize = 4;

    // * For every q_i, do FRI decommitment
    let fri_decommitment = fri_decommit_layers(&lde_fri_commitment, fri_decommitment_index);

    /*
        IMPORTANT NOTE:
        When we commit to the trace polynomial, let's call it f, we commit to an LDE of it.
        On the other hand, the fibonacci constraint (and in general, any constraint) related to f applies
        only using non-LDE roots of unity.
        In this case, the constraint is f(w^2 x) - f(w x) - f(x), where w is a 2^n root of unity.
        But for the commitment we use g, a 2^{nb} root of unity (b is the blowup factor).
        When we sample a value x to evaluate the trace polynomial on, it has to be a 2^{nb} root of unity,
        so with fiat-shamir we sample a random index in that range.
        When we provide evaluations, we provide them for x*(w^2), x*w and x.
    */

    let fri_layers_merkle_roots: Vec<FieldElement<F>> = lde_fri_commitment
        .iter()
        .map(|fri_commitment| fri_commitment.merkle_tree.root.clone())
        .collect();

    let ret = StarkQueryProof {
        trace_ood_frame_evaluations,
        composition_poly_evaluations,
        fri_layers_merkle_roots,
        fri_decommitment,
    };

    ret
}

/// Returns the DEEP composition polynomial that the prover then commits to using
/// FRI. This polynomial is a linear combination of the trace polynomial and the
/// composition polynomial, with coefficients sampled by the verifier (i.e. using Fiat-Shamir).
fn compute_deep_composition_poly<F: IsField>(
    trace_poly: &Polynomial<FieldElement<F>>,
    even_composition_poly: &Polynomial<FieldElement<F>>,
    odd_composition_poly: &Polynomial<FieldElement<F>>,
    ood_evaluation_point: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    // TODO: Fiat-Shamir
    let gamma_1 = FieldElement::<F>::one();
    let gamma_2 = FieldElement::<F>::one();
    let gamma_3 = FieldElement::<F>::one();
    let gamma_4 = FieldElement::<F>::one();
    let gamma_5 = FieldElement::<F>::one();
    let trace_term_coeffs = [gamma_1, gamma_2, gamma_3];

    // TODO: The frame_offsets argument is hard-coded for fibonacci here
    let trace_evaluations = Frame::get_trace_evaluations(
        &[trace_poly.clone()],
        &ood_evaluation_point,
        &[0, 1, 2],
        primitive_root,
    );

    // TODO: Hard-coded for fibonacci. We take the first element in  `trace_evaluations` because there is
    // only one transition constraint, but could be more for an arbitrary computation
    let mut trace_terms = Polynomial::zero();
    for (eval, coeff) in trace_evaluations[0].iter().zip(trace_term_coeffs) {
        let poly = (trace_poly.clone() - Polynomial::new_monomial(trace_poly.evaluate(&eval), 0))
            / (Polynomial::new_monomial(FieldElement::<F>::one(), 1)
                - Polynomial::new_monomial(eval.clone(), 0));

        trace_terms = trace_terms + poly * coeff;
    }

    // Evaluate in X^2
    let even_composition_poly = polynomial::compose(
        even_composition_poly,
        &Polynomial::new_monomial(FieldElement::one(), 2),
    );
    let odd_composition_poly = polynomial::compose(
        odd_composition_poly,
        &Polynomial::new_monomial(FieldElement::one(), 2),
    );

    let even_composition_poly_term = (even_composition_poly.clone()
        - Polynomial::new_monomial(
            even_composition_poly.evaluate(&ood_evaluation_point.clone()),
            0,
        ))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));

    let odd_composition_poly_term = (odd_composition_poly.clone()
        - Polynomial::new_monomial(odd_composition_poly.evaluate(ood_evaluation_point), 0))
        / (Polynomial::new_monomial(FieldElement::one(), 1)
            - Polynomial::new_monomial(ood_evaluation_point * ood_evaluation_point, 0));

    trace_terms + even_composition_poly_term * gamma_4 + odd_composition_poly_term * gamma_5
}

pub fn verify<F: IsField + IsTwoAdicField, A: AIR + AIR<Field = F>>(
    proof: &StarkQueryProof<F>,
    air: &A,
) -> bool {
    let transcript = &mut Transcript::new();

    // BEGIN TRACE <-> Composition poly consistency evaluation check

    let trace_poly_ood_frame_evaluations = &proof.trace_ood_frame_evaluations;

    // These are H_1(z^2) and H_2(z^2)
    let composition_poly_evaluations = &proof.composition_poly_evaluations;

    let trace_primitive_root =
        F::get_primitive_root_of_unity(air.context().trace_length as u64).unwrap();

    let boundary_constraints = air.compute_boundary_constraints();

    // TODO: Fiat-Shamir
    let z = FieldElement::<F>::from(3);

    // C_1(z)
    let domain = boundary_constraints.generate_roots_of_unity(&trace_primitive_root);
    // TODO: this assumes one column
    let values = boundary_constraints.values(0);

    // The boundary constraint polynomial is trace - this polynomial below.
    let boundary_interpolating_polynomial = &Polynomial::interpolate(&domain, &values);
    let boundary_zerofier = boundary_constraints.compute_zerofier(&trace_primitive_root);

    let boundary_alpha = FieldElement::<F>::one();
    let boundary_beta = FieldElement::<F>::one();

    let max_degree =
        air.context().trace_length * air.context().transition_degrees().iter().max().unwrap();

    let max_degree_power_of_two = helpers::next_power_of_two(max_degree as u64);

    // TODO: This is assuming one column
    let mut boundary_quotient_ood_evaluation = (&trace_poly_ood_frame_evaluations.get_row(0)[0]
        - boundary_interpolating_polynomial.evaluate(&z))
        / boundary_zerofier.evaluate(&z);

    // TODO: 31 is hardcoded here because we need to fix the way we take degrees anyway
    boundary_quotient_ood_evaluation = boundary_quotient_ood_evaluation
        * (&boundary_alpha * z.pow(max_degree_power_of_two - 31) + &boundary_beta);

    let transition_ood_frame_evaluations = air.compute_transition(trace_poly_ood_frame_evaluations);

    // TODO: Fiat-Shamir
    let alpha = FieldElement::one();
    let beta = FieldElement::one();

    let alpha_and_beta_transition_coefficients = vec![(alpha.clone(), beta.clone())];

    let c_i_evaluations = ConstraintEvaluator::compute_transition_evaluations(
        air,
        &transition_ood_frame_evaluations,
        &alpha_and_beta_transition_coefficients,
        max_degree_power_of_two,
        &z,
    );

    let composition_poly_ood_evaluation = &boundary_quotient_ood_evaluation
        + c_i_evaluations
            .iter()
            .fold(FieldElement::<F>::zero(), |acc, evaluation| {
                acc + evaluation
            });

    let composition_poly_claimed_ood_evaluation =
        &composition_poly_evaluations[0] + &z * &composition_poly_evaluations[1];

    println!(
        "COMPOSITION POLY CLAIMED OOD EVALUATION {:?}",
        composition_poly_claimed_ood_evaluation
    );
    println!(
        "COMPOSITION POLY ACTUAL OOD EVALUATION {:?}",
        composition_poly_ood_evaluation
    );

    println!("BEFORE CONSISTENCY CHECK");
    if composition_poly_claimed_ood_evaluation != composition_poly_ood_evaluation {
        return false;
    }
    println!("AFTER CONSISTENCY CHECK");

    // // END TRACE <-> Composition poly consistency evaluation check

    fri_verify(
        &proof.fri_layers_merkle_roots,
        &proof.fri_decommitment,
        transcript,
    )
}

/// Performs FRI verification for some decommitment
pub fn fri_verify<F: IsField + IsTwoAdicField>(
    fri_layers_merkle_roots: &[FieldElement<F>],
    fri_decommitment: &FriDecommitment<F>,
    _transcript: &mut Transcript,
) -> bool {
    // For each fri layer merkle proof check:
    // That each merkle path verifies

    // Sample beta with fiat shamir
    // Compute v = [P_i(z_i) + P_i(-z_i)] / 2 + beta * [P_i(z_i) - P_i(-z_i)] / (2 * z_i)
    // Where P_i is the folded polynomial of the i-th fiat shamir round
    // z_i is obtained from the first z (that was derived through fiat-shamir) through a known calculation
    // The calculation is, given the index, index % length_of_evaluation_domain

    // Check that v = P_{i+1}(z_i)

    // TODO: Fiat-Shamir
    let decommitment_index: u64 = 4;

    // let mut lde_primitive_root = generate_primitive_root(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE);
    let mut lde_primitive_root =
        F::get_primitive_root_of_unity(ORDER_OF_ROOTS_OF_UNITY_FOR_LDE).unwrap();
    let mut offset = FieldElement::from(COSET_OFFSET);

    // For each (merkle_root, merkle_auth_path) / fold
    // With the auth path containining the element that the
    // path proves it's existance
    for (
        index,
        (
            layer_number,
            (
                fri_layer_merkle_root,
                (
                    (fri_layer_auth_path, fri_layer_auth_path_symmetric),
                    (auth_path_evaluation, auth_path_evaluation_symmetric),
                ),
            ),
        ),
    ) in fri_layers_merkle_roots
        .iter()
        .zip(
            fri_decommitment
                .layer_merkle_paths
                .iter()
                .zip(fri_decommitment.layer_evaluations.iter()),
        )
        .enumerate()
        // Since we always derive the current layer from the previous layer
        // We start with the second one, skipping the first, so previous is layer is the first one
        .skip(1)
        .enumerate()
    {
        // This is the current layer's evaluation domain length. We need it to know what the decommitment index for the current
        // layer is, so we can check the merkle paths at the right index.
        let current_layer_domain_length = ORDER_OF_ROOTS_OF_UNITY_FOR_LDE >> layer_number;

        let layer_evaluation_index = decommitment_index % current_layer_domain_length;
        if !fri_layer_auth_path.verify(
            fri_layer_merkle_root,
            layer_evaluation_index as usize,
            auth_path_evaluation,
        ) {
            return false;
        }

        let layer_evaluation_index_symmetric =
            (decommitment_index + current_layer_domain_length) % current_layer_domain_length;

        if !fri_layer_auth_path_symmetric.verify(
            fri_layer_merkle_root,
            layer_evaluation_index_symmetric as usize,
            auth_path_evaluation_symmetric,
        ) {
            return false;
        }

        // TODO: Fiat Shamir
        // let beta = beta_list[index].clone();
        let beta = 1;

        let (previous_auth_path_evaluation, previous_path_evaluation_symmetric) = fri_decommitment
            .layer_evaluations
            .get(layer_number - 1)
            // TODO: Check at the start of the FRI operation
            // if layer_merkle_paths has the right amount of elements
            .unwrap();

        // evaluation point = offset * w ^ i in the Stark literature
        let evaluation_point = &offset * lde_primitive_root.pow(decommitment_index);

        // v is the calculated element for the
        // co linearity check
        let two = &FieldElement::from(2);
        let beta = FieldElement::from(beta);
        let v = (previous_auth_path_evaluation + previous_path_evaluation_symmetric) / two
            + &beta * (previous_auth_path_evaluation - previous_path_evaluation_symmetric)
                / (two * evaluation_point);

        lde_primitive_root = lde_primitive_root.pow(2_usize);
        offset = offset.pow(2_usize);

        if v != *auth_path_evaluation {
            return false;
        }

        // On the last iteration, also check the provided last evaluation point.
        if layer_number == fri_layers_merkle_roots.len() - 1 {
            let last_evaluation_point = &offset * lde_primitive_root.pow(decommitment_index);

            let last_v = (auth_path_evaluation + auth_path_evaluation_symmetric) / two
                + &beta * (auth_path_evaluation - auth_path_evaluation_symmetric)
                    / (two * &last_evaluation_point);

            if last_v != fri_decommitment.last_layer_evaluation {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        air::{
            constraints::boundary::BoundaryConstraint,
            context::{AirContext, ProofOptions},
        },
        verify, PrimeField, FE,
    };
    use air::constraints::boundary::BoundaryConstraints;

    use super::prove;
    use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U256, U384};

    #[derive(Clone)]
    pub struct FibonacciAIR {
        context: AirContext,
        trace: TraceTable<PrimeField>,
    }

    impl AIR for FibonacciAIR {
        type Field = PrimeField;

        fn new(trace: TraceTable<Self::Field>, context: air::context::AirContext) -> Self {
            Self {
                context: context,
                trace: trace,
            }
        }

        fn compute_transition(
            &self,
            frame: &air::frame::Frame<Self::Field>,
        ) -> Vec<FieldElement<Self::Field>> {
            let first_row = frame.get_row(0);
            let second_row = frame.get_row(1);
            let third_row = frame.get_row(2);

            vec![third_row[0].clone() - second_row[0].clone() - first_row[0].clone()]
        }

        fn compute_boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
            let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());
            let result = BoundaryConstraint::new_simple(7, FieldElement::<Self::Field>::from(21));

            BoundaryConstraints::from_constraints(vec![a0, a1, result])
        }

        fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
            let one_field = FieldElement::<Self::Field>::one();
            let roots_of_unity = Self::Field::get_powers_of_primitive_root_coset(
                self.context().trace_length as u64,
                self.context().trace_length,
                &one_field,
            )
            .unwrap();

            let mut result = vec![];

            for index in 0..self.context().num_transition_constraints {
                result.push(Polynomial::new_monomial(one_field.clone(), 0));

                for exemption_index in self.context().transition_exemptions {
                    result[index] = result[index].clone()
                        * (Polynomial::new_monomial(one_field.clone(), 1)
                            - Polynomial::new_monomial(roots_of_unity[exemption_index].clone(), 0));
                }
            }

            result
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }

    #[test]
    fn test_prove() {
        let trace = fibonacci_trace([FE::new(U256::from("1")), FE::new(U256::from("1"))]);

        let context = AirContext {
            options: ProofOptions { blowup_factor: 2 },
            trace_length: 32,
            trace_info: (32, 1),
            transition_degrees: vec![1],
            transition_exemptions: vec![30, 31],
            transition_offsets: vec![0, 1, 2],
            num_assertions: 3,
            num_transition_constraints: 1,
        };

        let trace = TraceTable {
            table: trace,
            num_cols: 1,
        };

        let fibonacci_air = FibonacciAIR::new(trace, context);

        let trace = fibonacci_trace([FE::new(U256::from("1")), FE::new(U256::from("1"))]);

        let result = prove(&trace, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;

    pub(crate) fn fibonacci_trace(initial_values: [FE; 2], iters: usize) -> Vec<FE> {
        let mut ret: Vec<FE> = vec![];

        ret.push(initial_values[0].clone());
        ret.push(initial_values[1].clone());

        for i in 2..iters {
            ret.push(ret[i - 1].clone() + ret[i - 2].clone());
        }

        ret
    }
}
