use lambdaworks_math::{
    circle::{
        cosets::Coset,
        point::CirclePoint,
    },
    field::fields::mersenne31::field::Mersenne31Field,
};

use super::air::AIR;

pub struct Domain {
    pub(crate) trace_length: usize,
    pub(crate) trace_log_2_length: u32,
    pub(crate) blowup_factor: usize,
    pub(crate) trace_coset_points: Vec<CirclePoint<Mersenne31Field>>,
    pub(crate) lde_coset_points: Vec<CirclePoint<Mersenne31Field>>,
    pub(crate) trace_group_generator: CirclePoint<Mersenne31Field>,
}

impl Domain {
    pub fn new<A>(air: &A) -> Self
    where
        A: AIR,
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
