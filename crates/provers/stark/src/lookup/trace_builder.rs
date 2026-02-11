use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::{prover::ProvingError, trace::TraceTable};

use super::types::{
    BusInteraction, LinearTerm, Multiplicity, LOGUP_CHALLENGE_ALPHA, LOGUP_CHALLENGE_Z,
};

/// Builds a term column for a bus interaction.
///
/// Each row contains: `term[i] = sign * multiplicity[i] / fingerprint[i]`
///
/// where:
/// - `fingerprint[i] = z - (bus_id*α^0 + v[0]*α^1 + v[1]*α^2 + ...)`
/// - `sign = +1` for senders, `-1` for receivers
///
/// Returns an error if any fingerprint evaluates to zero (astronomically
/// unlikely for randomly sampled challenges, probability ≈ N/|F|).
pub fn build_logup_term_column<F, E>(
    aux_column_idx: usize,
    interaction: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace: &mut TraceTable<F, E>,
    challenges: &[FieldElement<E>],
) -> Result<(), ProvingError>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let trace_len = trace.num_rows();

    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    // Precompute powers of alpha
    let num_bus_elements = interaction.num_bus_elements();
    let alpha_powers: Vec<FieldElement<E>> = (0..num_bus_elements).map(|i| alpha.pow(i)).collect();

    let sign = if interaction.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    for row in 0..trace_len {
        // Compute multiplicity
        let multiplicity: FieldElement<F> =
            compute_trace_multiplicity(main_segment_cols, row, &interaction.multiplicity);

        // Bus elements: [bus_id, ...values...]
        let mut bus_elements: Vec<FieldElement<E>> = vec![FieldElement::from(interaction.bus_id)];

        bus_elements.extend(interaction.values.iter().map(|bv| {
            let combined: FieldElement<F> =
                bv.combine_from(|col| main_segment_cols[col][row].clone());
            combined.to_extension()
        }));

        // fingerprint = z - (bus_id*α^0 + v[0]*α^1 + ...)
        let linear_combination: FieldElement<E> = bus_elements
            .iter()
            .zip(alpha_powers.iter())
            .map(|(v, coeff)| v * coeff)
            .sum();

        let fingerprint = z - &linear_combination;

        // term = sign * multiplicity / fingerprint
        let fingerprint_inv = fingerprint.inv().map_err(|_| {
            ProvingError::WrongParameter(format!(
                "LogUp: zero fingerprint at row {row} for bus_id {}. \
                 Try re-proving — this is astronomically unlikely (≈ 1/|F|).",
                interaction.bus_id,
            ))
        })?;

        let term = multiplicity * &sign * fingerprint_inv;
        trace.set_aux(row, aux_column_idx, term);
    }

    Ok(())
}

/// Builds the accumulated column that sums all term columns across rows.
///
/// `acc[0] = Σ term_columns[0]`
/// `acc[i] = acc[i-1] + Σ term_columns[i]`
pub fn build_accumulated_column<F, E>(
    acc_column_idx: usize,
    num_term_columns: usize,
    trace: &mut TraceTable<F, E>,
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let trace_len = trace.num_rows();
    let mut accumulated = FieldElement::<E>::zero();

    for row in 0..trace_len {
        let mut row_sum = FieldElement::<E>::zero();
        for term_col in 0..num_term_columns {
            row_sum += trace.get_aux(row, term_col).clone();
        }
        accumulated += row_sum;
        trace.set_aux(row, acc_column_idx, accumulated.clone());
    }
}

/// Compute multiplicity from main trace columns for a given row.
fn compute_trace_multiplicity<F: IsField>(
    main_cols: &[Vec<FieldElement<F>>],
    row: usize,
    multiplicity: &Multiplicity,
) -> FieldElement<F> {
    match multiplicity {
        Multiplicity::One => FieldElement::<F>::one(),
        Multiplicity::Column(col) => main_cols[*col][row].clone(),
        Multiplicity::Sum(col_a, col_b) => &main_cols[*col_a][row] + &main_cols[*col_b][row],
        Multiplicity::Negated(col) => FieldElement::<F>::one() - &main_cols[*col][row],
        Multiplicity::Linear(terms) => {
            let mut result = FieldElement::<F>::zero();
            for term in terms {
                match term {
                    LinearTerm::Column {
                        coefficient,
                        column,
                    } => {
                        let coeff = if *coefficient >= 0 {
                            FieldElement::<F>::from(*coefficient as u64)
                        } else {
                            -FieldElement::<F>::from((-*coefficient) as u64)
                        };
                        result += &main_cols[*column][row] * coeff;
                    }
                    LinearTerm::ColumnUnsigned {
                        coefficient,
                        column,
                    } => {
                        let coeff = FieldElement::<F>::from(*coefficient);
                        result += &main_cols[*column][row] * coeff;
                    }
                    LinearTerm::Constant(value) => {
                        if *value >= 0 {
                            result += FieldElement::<F>::from(*value as u64);
                        } else {
                            result = result - FieldElement::<F>::from((-*value) as u64);
                        }
                    }
                    LinearTerm::ConstantUnsigned(value) => {
                        result += FieldElement::<F>::from(*value);
                    }
                }
            }
            result
        }
    }
}
