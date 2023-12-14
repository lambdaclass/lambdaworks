use std::ops::Div;

use crate::frame::Frame;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use num_integer::Integer;

pub trait TransitionConstraint<F: IsFFTField> {
    fn degree(&self) -> usize;

    fn constraint_index(&self) -> usize;

    fn evaluate(
        &self,
        frame: &Frame<F>,
        transition_evaluations: &mut [FieldElement<F>],
        periodic_values: &[FieldElement<F>],
        rap_challenges: &[FieldElement<F>],
    );

    fn period(&self) -> usize {
        1
    }

    fn exemptions_period(&self) -> Option<usize> {
        None
    }

    fn periodic_exemptions_offset(&self) -> Option<usize> {
        None
    }

    fn end_exemptions(&self) -> usize;

    fn zerofier_evaluations(
        &self,
        blowup_factor: usize,
        offset: &FieldElement<F>,
        trace_length: usize,
        trace_primitive_root: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        let root_order = (blowup_factor * trace_length).trailing_zeros();
        let root = F::get_primitive_root_of_unity(root_order).unwrap();
        let one = FieldElement::<F>::one();

        if let Some(exemptions_period) = self.exemptions_period() {
            // FIXME: Rather than making this assertions here, it would be better to handle these errors or
            // make these checks when the AIR is initialized.
            debug_assert!(self.period().is_multiple_of(exemptions_period));
            debug_assert!(self.periodic_exemptions_offset().is_some());

            let last_exponent = blowup_factor * exemptions_period;

            (0..last_exponent)
                .map(|exponent| {
                    let x = root.pow(exponent);
                    let offset_times_x = offset * x;
                    let numerator = offset_times_x.pow(trace_length / self.period()) - &one;
                    let offset_exponent = trace_length * self.periodic_exemptions_offset().unwrap()
                        / exemptions_period;

                    numerator.div(
                        offset_times_x.pow(trace_length / exemptions_period)
                            - trace_primitive_root.pow(offset_exponent),
                    )
                })
                .collect();
        } else {
            let last_exponent = blowup_factor * self.period();

            (0..last_exponent).map(|exponent| {
                let x = root.pow(exponent);
                (offset * x).pow(trace_length / self.period()) - &one
            })
        }
    }
}
