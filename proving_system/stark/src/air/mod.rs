use self::{
    constraints::boundary::BoundaryConstraints,
    context::{AirContext, ProofOptions},
    frame::Frame,
    trace::TraceTable,
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_fft::roots_of_unity::get_powers_of_primitive_root_coset;
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    polynomial::Polynomial,
};

pub mod constraints;
pub mod context;
#[cfg(debug_assertions)]
pub mod debug;
pub mod example;
pub mod frame;
pub mod trace;

pub trait AIR: Clone {
    type RawTrace;
    type RAPChallenges;

    fn build_main_trace(raw_trace: &Self::RawTrace) -> TraceTable;

    fn build_auxiliary_trace(
        main_trace: &TraceTable,
        rap_challenges: &Self::RAPChallenges,
    ) -> TraceTable;

    fn build_rap_challenges<T: Transcript>(transcript: &mut T) -> Self::RAPChallenges;

    fn number_auxiliary_rap_columns(&self) -> usize {
        0
    }

    fn compute_transition(
        &self,
        frame: &Frame<Stark252PrimeField>,
        rap_challenges: &Self::RAPChallenges,
    ) -> Vec<FieldElement<Stark252PrimeField>>;
    fn boundary_constraints(
        &self,
        rap_challenges: &Self::RAPChallenges,
    ) -> BoundaryConstraints<Stark252PrimeField>;
    fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Stark252PrimeField>>> {
        let trace_length = self.context().trace_length;
        let roots_of_unity_order = trace_length.trailing_zeros();
        let roots_of_unity = get_powers_of_primitive_root_coset(
            roots_of_unity_order as u64,
            self.context().trace_length,
            &FieldElement::<Stark252PrimeField>::one(),
        )
        .unwrap();

        let mut result = vec![];
        let x_n = Polynomial::new_monomial(FieldElement::one(), trace_length);
        let x = Polynomial::new_monomial(FieldElement::one(), 1);
        for transition_idx in 0..self.context().num_transition_constraints {
            // X^(trace_length) - 1
            let roots_of_unity_vanishing_polynomial = &x_n - FieldElement::one();

            let mut exemptions_polynomial =
                Polynomial::new_monomial(FieldElement::<Stark252PrimeField>::one(), 0);

            for i in 0..self.context().transition_exemptions[transition_idx] {
                exemptions_polynomial =
                    exemptions_polynomial * (&x - &roots_of_unity[roots_of_unity.len() - 1 - i])
            }

            result.push(roots_of_unity_vanishing_polynomial / exemptions_polynomial);
        }

        result
    }
    fn context(&self) -> AirContext;
    fn options(&self) -> ProofOptions {
        self.context().options
    }

    fn blowup_factor(&self) -> u8 {
        self.options().blowup_factor
    }

    fn num_transition_constraints(&self) -> usize {
        self.context().num_transition_constraints
    }
}
