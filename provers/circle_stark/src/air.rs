use super::{constraints::boundary::BoundaryConstraints, frame::Frame};
use crate::{
    air_context::AirContext, constraints::transition::TransitionConstraint, domain::Domain,
};
use lambdaworks_math::{
    circle::point::CirclePoint,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};
use std::collections::HashMap;
type ZerofierGroupKey = (usize, usize, Option<usize>, Option<usize>, usize);
/// AIR is a representation of the Constraints
pub trait AIR {
    type Field: IsFFTField + IsSubFieldOf<Self::FieldExtension> + Send + Sync;
    type FieldExtension: IsField + Send + Sync;
    type PublicInputs;

    fn new(trace_length: usize, pub_inputs: &Self::PublicInputs) -> Self;

    /// Returns the amount trace columns.
    fn trace_layout(&self) -> usize;

    fn composition_poly_degree_bound(&self) -> usize;

    /// The method called by the prover to evaluate the transitions corresponding to an evaluation frame.
    /// In the case of the prover, the main evaluation table of the frame takes values in
    /// `Self::Field`, since they are the evaluations of the main trace at the LDE domain.
    fn compute_transition_prover(
        &self,
        frame: &Frame<Self::Field>,
    ) -> Vec<FieldElement<Self::Field>> {
        let mut evaluations =
            vec![FieldElement::<Self::Field>::zero(); self.num_transition_constraints()];
        self.transition_constraints()
            .iter()
            .for_each(|c| c.evaluate(frame, &mut evaluations));
        evaluations
    }

    fn boundary_constraints(&self) -> BoundaryConstraints<Self::FieldExtension>;

    /// The method called by the verifier to evaluate the transitions at the out of domain frame.
    /// In the case of the verifier, both main and auxiliary tables of the evaluation frame take
    /// values in `Self::FieldExtension`, since they are the evaluations of the trace polynomials
    /// at the out of domain challenge.
    /// In case `Self::Field` coincides with `Self::FieldExtension`, this method and
    /// `compute_transition_prover` should return the same values.
    fn compute_transition_verifier(
        &self,
        frame: &Frame<Self::Field>,
    ) -> Vec<FieldElement<Self::FieldExtension>>;

    fn context(&self) -> &AirContext;

    fn trace_length(&self) -> usize;

    fn blowup_factor(&self) -> u8 {
        2
    }

    fn trace_group_generator(&self) -> CirclePoint<Self::Field> {
        let trace_length = self.trace_length();
        let log_2_length = trace_length.trailing_zeros();
        CirclePoint::get_generator_of_subgroup(log_2_length)
    }

    fn num_transition_constraints(&self) -> usize {
        self.context().num_transition_constraints
    }

    fn pub_inputs(&self) -> &Self::PublicInputs;

    fn transition_constraints(
        &self,
    ) -> &Vec<Box<dyn TransitionConstraint<Self::Field, Self::FieldExtension>>>;

    fn transition_zerofier_evaluations(
        &self,
        domain: &Domain<Self::Field>,
    ) -> Vec<Vec<FieldElement<Self::Field>>> {
        let mut evals = vec![Vec::new(); self.num_transition_constraints()];
        let mut zerofier_groups: HashMap<ZerofierGroupKey, Vec<FieldElement<Self::Field>>> =
            HashMap::new();

        self.transition_constraints().iter().for_each(|c| {
            let end_exemptions = c.end_exemptions();
            // This hashmap is used to avoid recomputing with an fft the same zerofier evaluation
            // If there are multiple domain and subdomains it can be further optimized
            // as to share computation between them
            let zerofier_group_key = (end_exemptions);
            zerofier_groups
                .entry(zerofier_group_key)
                .or_insert_with(|| c.zerofier_evaluations_on_extended_domain(domain));
            let zerofier_evaluations = zerofier_groups.get(&zerofier_group_key).unwrap();
            evals[c.constraint_idx()] = zerofier_evaluations.clone();
        });
        evals
    }
}
