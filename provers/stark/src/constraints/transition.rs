use std::iter::Cycle;
use std::ops::Div;
use std::vec::IntoIter;

use crate::domain::Domain;
use crate::frame::Frame;
use crate::prover::evaluate_polynomial_on_lde_domain;
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

    fn end_exemptions_poly(
        &self,
        trace_primitive_root: &FieldElement<F>,
        trace_length: usize,
    ) -> Polynomial<FieldElement<F>> {
        let one_poly = Polynomial::new_monomial(FieldElement::<F>::one(), 0);
        if self.end_exemptions() == 0 {
            return one_poly;
        }

        let period = self.period();
        (1..=self.end_exemptions())
            .map(|exemption| trace_primitive_root.pow(trace_length - exemption * period))
            .fold(one_poly, |acc, offset| {
                acc * (Polynomial::new_monomial(FieldElement::<F>::one(), 1) - offset)
            })
    }

    fn zerofier_evaluations_on_extended_domain(&self, domain: &Domain<F>) -> Vec<FieldElement<F>> {
        let blowup_factor = domain.blowup_factor;
        let trace_length = domain.trace_roots_of_unity.len();
        let trace_primitive_root = &domain.trace_primitive_root;
        let coset_offset = &domain.coset_offset;

        let root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
        let root = F::get_primitive_root_of_unity(root_order).unwrap();

        println!("OMEGA TO THE N POWER: {:?}", root.pow(trace_length));

        println!(
            "ROOT TO THE NxBETA POWER: {:?}",
            root.pow(trace_length * blowup_factor)
        );

        println!("OMEGA TO THE BETA POWER: {:?}", root.pow(blowup_factor));

        let end_exemptions_poly = self.end_exemptions_poly(trace_primitive_root, trace_length);

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

            let mut evaluations: Vec<_> = (0..last_exponent)
                .map(|exponent| {
                    let x = root.pow(exponent);
                    (coset_offset * &x).pow(trace_length / self.period()) - FieldElement::<F>::one()
                })
                .collect();

            FieldElement::inplace_batch_inverse(&mut evaluations).unwrap();

            println!("ZEROFIER EVALS");
            for (i, eval) in evaluations.iter().enumerate() {
                println!("ZEROFIER EVAL {} - {:?}", i, eval);
            }

            let end_exemption_evaluations = evaluate_polynomial_on_lde_domain(
                &end_exemptions_poly,
                blowup_factor,
                domain.interpolation_domain_size,
                &coset_offset,
            )
            .unwrap();

            let cycled_evaluations = evaluations
                .iter()
                .cycle()
                .take(end_exemption_evaluations.len());

            std::iter::zip(cycled_evaluations, end_exemption_evaluations)
                .map(|(eval, exemption_eval)| eval * exemption_eval)
                .collect()
        }
    }
}

pub(crate) struct TransitionZerofiersIter<F: IsFFTField> {
    num_constraints: usize,
    zerofier_evals: Vec<IntoIter<FieldElement<F>>>,
}

impl<F> TransitionZerofiersIter<F>
where
    F: IsFFTField,
{
    pub(crate) fn new(zerofier_evals: Vec<Vec<FieldElement<F>>>) -> Self {
        let first_evals_len = zerofier_evals[0].len();
        debug_assert!(zerofier_evals
            .iter()
            .all(|evals| evals.len() == first_evals_len));

        let num_constraints = zerofier_evals.len();
        let zerofier_evals = zerofier_evals
            .into_iter()
            .map(|evals| evals.into_iter())
            .collect();

        Self {
            num_constraints,
            zerofier_evals,
        }
    }
}

impl<F> Iterator for TransitionZerofiersIter<F>
where
    F: IsFFTField,
{
    type Item = Vec<FieldElement<F>>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_evals = self
            .zerofier_evals
            .iter_mut()
            .map(|evals| evals.next().unwrap())
            .collect();

        Some(next_evals)
    }
}