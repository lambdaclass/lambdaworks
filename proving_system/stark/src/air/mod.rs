use self::{
    constraints::boundary::BoundaryConstraints,
    context::{AirContext, ProofOptions},
    frame::Frame,
    trace::{AuxSegmentInfo, AuxiliarySegment, TraceTable},
};
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
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

#[derive(Clone, Debug)]
pub struct TraceLayout {
    pub main_segment_width: usize,
    pub info: Option<AuxSegmentInfo>,
}

#[derive(Clone, Debug)]
pub struct TraceInfo {
    pub layout: TraceLayout,
    pub trace_length: usize,
}

pub trait AIR: Clone {
    type Field: IsFFTField;

    fn compute_transition(&self, frame: &Frame<Self::Field>) -> Vec<FieldElement<Self::Field>>;
    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field>;
    fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
        let trace_length = self.trace_length();
        let roots_of_unity_order = trace_length.trailing_zeros();
        let roots_of_unity = get_powers_of_primitive_root_coset(
            roots_of_unity_order as u64,
            trace_length,
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
    fn trace_info(&self) -> TraceInfo {
        self.context().trace_info
    }
    fn trace_length(&self) -> usize {
        self.trace_info().trace_length
    }
    fn num_aux_segments(&self) -> usize {
        if let Some(info) = self.trace_info().layout.info {
            info.num_aux_segments
        } else {
            0
        }
    }
    fn aux_segment_rand_coeffs<T: Transcript>(
        &self,
        segment_idx: usize,
        transcript: &T,
    ) -> Option<Vec<FieldElement<Self::Field>>> {
        None
    }

    #[allow(unused)]
    fn build_aux_segment(
        &self,
        trace: &TraceTable<Self::Field>,
        rand_elements: &[FieldElement<Self::Field>],
    ) -> Option<AuxiliarySegment<Self::Field>> {
        None
    }
}
