use lambdaworks_math::{
    circle::{
        cosets::Coset,
        point::{CirclePoint, HasCircleParams},
    },
    field::{element::FieldElement, traits::IsFFTField},
};

use super::air::AIR;

pub struct Domain<F: IsFFTField + HasCircleParams<F>> {
    pub(crate) trace_length: usize,
    pub(crate) trace_log_2_length: u32,
    pub(crate) blowup_factor: usize,
    pub(crate) trace_coset_points: Vec<CirclePoint<F>>,
    pub(crate) lde_coset_points: Vec<CirclePoint<F>>,
    pub(crate) trace_group_generator: CirclePoint<F>,
}

impl<F: IsFFTField + HasCircleParams<F>> Domain<F> {
    pub fn new<A>(air: &A) -> Self
    where
        A: AIR<Field = F>,
    {
        // Initial definitions
        let trace_length = air.trace_length();
        let trace_log_2_length = trace_length.trailing_zeros();
        let blowup_factor = air.blowup_factor() as usize;

        // * Generate Coset
        let trace_coset_points = Coset::get_coset_points(&Coset::new_standard(trace_log_2_length));
        let lde_coset_points = Coset::get_coset_points(&Coset::new_standard(
            (blowup_factor * trace_length).trailing_zeros(),
        ));
        let trace_group_generator = CirclePoint::get_generator_of_subgroup(trace_log_2_length);

        Self {
            trace_length,
            trace_log_2_length,
            blowup_factor,
            trace_coset_points,
            lde_coset_points,
            trace_group_generator,
        }
    }
}
