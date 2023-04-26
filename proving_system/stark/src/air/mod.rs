use self::{
    constraints::boundary::BoundaryConstraints,
    context::{AirContext, ProofOptions},
    frame::Frame,
};
use lambdaworks_fft::roots_of_unity::get_powers_of_primitive_root_coset;
use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

pub mod constraints;
pub mod context;
pub mod example;
pub mod frame;
pub mod trace;

pub trait AIR: Clone {
    type Field: IsFFTField;

    fn compute_transition(&self, frame: &Frame<Self::Field>) -> Vec<FieldElement<Self::Field>>;
    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field>;
    fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let trace_length = self.context().trace_length;
        let roots_of_unity_order = trace_length.trailing_zeros();
        let roots_of_unity = get_powers_of_primitive_root_coset(
            roots_of_unity_order as u64,
            self.context().trace_length,
            &FieldElement::<Self::Field>::one(),
        )
        .unwrap();

        let mut result = vec![];
        for transition_idx in 0..self.context().num_transition_constraints {
            // X^(trace_length) - 1
            let roots_of_unity_vanishing_polynomial =
                Polynomial::new_monomial(FieldElement::<Self::Field>::one(), trace_length)
                    - Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 0);

            let mut exemptions_polynomial =
                Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 0);

            for i in 0..self.context().transition_exemptions[transition_idx] {
                exemptions_polynomial = exemptions_polynomial
                    * (Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 1)
                        - Polynomial::new_monomial(
                            roots_of_unity[roots_of_unity.len() - 1 - i].clone(),
                            0,
                        ))
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
