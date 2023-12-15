use std::iter::Cycle;
use std::ops::Div;
use std::slice::Iter;

use crate::frame::Frame;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;
use num_integer::Integer;

pub trait TransitionConstraint<F: IsFFTField> {
    fn degree(&self) -> usize;

    fn constraint_idx(&self) -> usize;

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
        let root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
        let root = F::get_primitive_root_of_unity(root_order).unwrap();
        let one = FieldElement::<F>::one();

        let end_exemptions_poly = if self.end_exemptions() == 0 {
            Polynomial::new(&[one])
        } else {
            let period = self.period();
            let one_poly = Polynomial::new_monomial(FieldElement::<F>::one(), 0);
            (1..self.end_exemptions())
                // .rev()
                .map(|exemption| trace_primitive_root.pow(trace_length - exemption * period))
                .fold(one_poly, |acc, offset| {
                    acc * Polynomial::new_monomial(one, trace_length / period)
                })
        };

        // In the first branch of this if statement, the evaluations are calculated directly
        // by computing P_exemptions(x) / Zerofier(x)
        if let Some(exemptions_period) = self.exemptions_period() {
            // FIXME: Rather than making this assertions here, it would be better to handle these
            // errors or make these checks when the AIR is initialized.
            debug_assert!(self.period().is_multiple_of(&exemptions_period));
            debug_assert!(self.periodic_exemptions_offset().is_some());

            let last_exponent = blowup_factor * exemptions_period;

            (0..last_exponent)
                .map(|exponent| {
                    let x = root.pow(exponent);
                    let offset_times_x = offset * x;
                    let offset_exponent = trace_length * self.periodic_exemptions_offset().unwrap()
                        / exemptions_period;

                    let denominator = offset_times_x.pow(trace_length / self.period()) - &one;
                    let numerator = offset_times_x.pow(trace_length / exemptions_period)
                        - trace_primitive_root.pow(offset_exponent);

                    numerator.div(denominator) * end_exemptions_poly.evaluate(&x)
                })
                .collect()
        // In this else branch, the zerofiers are computed as the numerator, then inverted
        // using batch inverse and then multiplied by P_exemptions(x). This way we don't do
        // useless divisions.
        } else {
            let last_exponent = blowup_factor * self.period();

            let (mut evaluations, xs): (Vec<_>, Vec<_>) = (0..last_exponent)
                .map(|exponent| {
                    let x = root.pow(exponent);
                    let eval = (offset * x).pow(trace_length / self.period()) - &one;
                    (eval, x)
                })
                .unzip();

            FieldElement::inplace_batch_inverse(&mut evaluations);

            evaluations
                .iter()
                .zip(xs)
                .map(|(eval, x)| eval * end_exemptions_poly.evaluate(&x))
                .collect()
        }
    }
}

pub(crate) struct TransitionZerofiersIter<'z, F: IsFFTField> {
    num_constraints: usize,
    zerofier_eval_cycles: Vec<Cycle<Iter<'z, FieldElement<F>>>>,
}

impl<'z, F> TransitionZerofiersIter<'z, F>
where
    F: IsFFTField,
{
    pub(crate) fn new(zerofier_evals: Vec<Vec<FieldElement<F>>>) -> Self {
        let num_constraints = zerofier_evals.len();
        let zerofier_eval_cycles = zerofier_evals
            .into_iter()
            .map(|evals| evals.iter().cycle())
            .collect();

        Self {
            num_constraints,
            zerofier_eval_cycles,
        }
    }
}

impl<'z, F> Iterator for TransitionZerofiersIter<'z, F>
where
    F: IsFFTField,
{
    type Item = Vec<&'z FieldElement<F>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_evals_cycle = self
            .zerofier_eval_cycles
            .iter()
            .map(|eval_cycle| eval_cycle.next().unwrap())
            .collect();

        Some(next_evals_cycle)
    }
}
