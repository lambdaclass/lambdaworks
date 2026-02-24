use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use crate::prover::ProvingError;

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
/// Uses batch inversion: all fingerprints are collected first, then inverted
/// in a single pass using Montgomery's trick (one field inversion + O(N)
/// multiplications instead of O(N) inversions).
///
/// Returns an error if any fingerprint evaluates to zero (astronomically
/// unlikely for randomly sampled challenges, probability ≈ N/|F|).
pub fn build_logup_term_column<F, E>(
    interaction: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Result<Vec<FieldElement<E>>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    let z = &challenges[LOGUP_CHALLENGE_Z];
    let alpha = &challenges[LOGUP_CHALLENGE_ALPHA];

    // Iterative alpha powers: O(1) per power instead of O(log i) for alpha.pow(i)
    let num_bus_elements = interaction.num_bus_elements();
    let mut alpha_powers = Vec::with_capacity(num_bus_elements);
    let mut alpha_power = FieldElement::<E>::one();
    for _ in 0..num_bus_elements {
        alpha_powers.push(alpha_power.clone());
        alpha_power = &alpha_power * alpha;
    }

    let sign = if interaction.is_sender {
        FieldElement::<E>::one()
    } else {
        -FieldElement::<E>::one()
    };

    // Precompute bus_id in extension field (avoids per-row conversion)
    let bus_id_ext: FieldElement<E> = FieldElement::<F>::from(interaction.bus_id).to_extension();

    // First pass: compute all fingerprints (no per-row allocation)
    // Note: `row` indexes into multiple columns inside combine_from, so range loop is needed.
    let mut fingerprints: Vec<FieldElement<E>> = Vec::with_capacity(trace_len);
    #[allow(clippy::needless_range_loop)]
    for row in 0..trace_len {
        let mut linear_combination = &bus_id_ext * &alpha_powers[0];
        for (bv, alpha_pow) in interaction.values.iter().zip(alpha_powers[1..].iter()) {
            let combined: FieldElement<F> =
                bv.combine_from(|col| main_segment_cols[col][row].clone());
            linear_combination += &combined.to_extension() * alpha_pow;
        }
        fingerprints.push(z - &linear_combination);
    }

    // Batch inversion: one field inversion + O(N) multiplications
    FieldElement::inplace_batch_inverse(&mut fingerprints).map_err(|_| {
        ProvingError::WrongParameter(
            "LogUp: zero fingerprint. Try re-proving (probability ≈ 1/|F|).".to_string(),
        )
    })?;

    // Second pass: compute terms = sign * multiplicity * fingerprint_inv
    let column: Vec<FieldElement<E>> = fingerprints
        .iter()
        .enumerate()
        .map(|(row, fp_inv)| {
            let multiplicity: FieldElement<F> =
                compute_trace_multiplicity(main_segment_cols, row, &interaction.multiplicity);
            multiplicity * &sign * fp_inv
        })
        .collect();

    Ok(column)
}

/// Builds the accumulated column that sums all term columns across rows.
///
/// `acc[0] = Σ term_columns[0]`  (sum of all terms at row 0)
/// `acc[i] = acc[i-1] + Σ term_columns[i]`  for i > 0
///
/// Takes pre-computed term column vectors (column-major layout for cache-friendly
/// sequential access within each column).
pub fn build_accumulated_column<E: IsField>(
    term_columns: &[Vec<FieldElement<E>>],
) -> Vec<FieldElement<E>> {
    if term_columns.is_empty() {
        return vec![];
    }

    let trace_len = term_columns[0].len();
    let mut accumulated_col = Vec::with_capacity(trace_len);
    let mut accumulated = FieldElement::<E>::zero();

    for row in 0..trace_len {
        let mut row_sum = FieldElement::<E>::zero();
        for term_col in term_columns {
            row_sum += term_col[row].clone();
        }
        accumulated += row_sum;
        accumulated_col.push(accumulated.clone());
    }

    accumulated_col
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
                            -FieldElement::<F>::from(coefficient.unsigned_abs())
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
                            result = result - FieldElement::<F>::from(value.unsigned_abs());
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
