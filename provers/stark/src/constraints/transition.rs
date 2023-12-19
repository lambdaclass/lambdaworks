use std::iter::Cycle;
use std::ops::Div;
use std::vec::IntoIter;

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
        coset_offset: &FieldElement<F>,
        trace_length: usize,
        trace_primitive_root: &FieldElement<F>,
    ) -> Vec<FieldElement<F>> {
        let root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
        let root = F::get_primitive_root_of_unity(root_order).unwrap();

        println!("OMEGA TO THE N POWER: {:?}", root.pow(trace_length));

        println!(
            "ROOT TO THE NxBETA POWER: {:?}",
            root.pow(trace_length * blowup_factor)
        );

        println!("OMEGA TO THE BETA POWER: {:?}", root.pow(blowup_factor));

        let one_poly = Polynomial::new_monomial(FieldElement::<F>::one(), 0);
        let end_exemptions_poly = if self.end_exemptions() == 0 {
            one_poly
        } else {
            let period = self.period();
            (1..=self.end_exemptions())
                .map(|exemption| trace_primitive_root.pow(trace_length - exemption * period))
                .fold(one_poly, |acc, offset| {
                    acc * (Polynomial::new_monomial(FieldElement::<F>::one(), 1) - offset)
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
                    let offset_times_x = coset_offset * &x;
                    let offset_exponent = trace_length * self.periodic_exemptions_offset().unwrap()
                        / exemptions_period;

                    let denominator = offset_times_x.pow(trace_length / self.period())
                        - &FieldElement::<F>::one();
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
                    let eval = (coset_offset * &x).pow(trace_length / self.period())
                        - FieldElement::<F>::one();
                    (eval, x)
                })
                .unzip();

            FieldElement::inplace_batch_inverse(&mut evaluations).unwrap();

            println!("ZEROFIER EVALS");
            for (i, eval) in evaluations.iter().enumerate() {
                println!("ZEROFIER EVAL {} - {:?}", i, eval);
            }

            std::iter::zip(evaluations, xs)
                .map(|(eval, x)| eval * end_exemptions_poly.evaluate(&x))
                .collect()
        }
    }
}

pub(crate) struct TransitionZerofiersIter<F: IsFFTField> {
    num_constraints: usize,
    zerofier_eval_cycles: Vec<Cycle<IntoIter<FieldElement<F>>>>,
}

impl<F> TransitionZerofiersIter<F>
where
    F: IsFFTField,
{
    pub(crate) fn new(zerofier_evals: Vec<Vec<FieldElement<F>>>) -> Self {
        let num_constraints = zerofier_evals.len();
        let zerofier_eval_cycles = zerofier_evals
            .into_iter()
            .map(|evals| evals.into_iter().cycle())
            .collect();

        Self {
            num_constraints,
            zerofier_eval_cycles,
        }
    }
}

impl<F> Iterator for TransitionZerofiersIter<F>
where
    F: IsFFTField,
{
    type Item = Vec<FieldElement<F>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_evals_cycle = self
            .zerofier_eval_cycles
            .iter_mut()
            .map(|eval_cycle| eval_cycle.next().unwrap())
            .collect();

        Some(next_evals_cycle)
    }
}
