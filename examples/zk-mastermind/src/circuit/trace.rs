//! Trace table generation for ZK Mastermind
//!
//! The trace table layout (12 columns):
//! - Columns 0-3: Secret code (private witness)
//! - Columns 4-7: Guess code (public input)
//! - Columns 8-11: Auxiliary columns for calculations

use lambdaworks_math::field::{element::FieldElement, traits::IsFFTField};
use stark_platinum_prover::trace::TraceTable;

use crate::game::{Feedback, Guess, SecretCode};

/// Column indices matching the AIR definition
pub mod cols {
    pub const SECRET_0: usize = 0;
    #[allow(dead_code)]
    pub const SECRET_1: usize = 1;
    #[allow(dead_code)]
    pub const SECRET_2: usize = 2;
    #[allow(dead_code)]
    pub const SECRET_3: usize = 3;
    pub const GUESS_0: usize = 4;
    #[allow(dead_code)]
    pub const GUESS_1: usize = 5;
    #[allow(dead_code)]
    pub const GUESS_2: usize = 6;
    #[allow(dead_code)]
    pub const GUESS_3: usize = 7;
    pub const AUX_EXACT: usize = 8;
    pub const AUX_PARTIAL: usize = 9;
    pub const AUX_S_COUNT: usize = 10;
    pub const AUX_G_COUNT: usize = 11;
}

/// Generate a trace table for the Mastermind circuit
///
/// # Arguments
/// * `secret` - The secret code (private witness)
/// * `guess` - The guess (public input)
/// * `feedback` - The expected feedback
///
/// # Returns
/// A trace table with the computation trace
pub fn generate_trace<F: IsFFTField>(
    secret: &SecretCode,
    guess: &Guess,
    feedback: &Feedback,
) -> TraceTable<F, F> {
    // Trace length must be a power of 2 for FFT
    // We use 16 rows which is sufficient for our constraints
    let trace_length = 16;
    let num_cols = 12;

    // Create columns as vectors of field elements
    let mut columns: Vec<Vec<FieldElement<F>>> = vec![vec![]; num_cols];

    // Convert secret and guess to u64 values
    let secret_values: Vec<u64> = secret.0.iter().map(|c| *c as u64).collect();
    let guess_values: Vec<u64> = guess.0.iter().map(|c| *c as u64).collect();

    // Fill the trace table
    for row in 0..trace_length {
        // Columns 0-3: Secret code (repeated in all rows)
        for i in 0..4 {
            columns[i].push(FieldElement::from(secret_values[i]));
        }

        // Columns 4-7: Guess code (repeated in all rows)
        for i in 0..4 {
            columns[4 + i].push(FieldElement::from(guess_values[i]));
        }

        // Columns 8: Exact matches (feedback verification)
        let exact = FieldElement::from(feedback.exact as u64);
        columns[cols::AUX_EXACT].push(exact);

        // Columns 9: Partial matches
        let partial = FieldElement::from(feedback.partial as u64);
        columns[cols::AUX_PARTIAL].push(partial);

        // Columns 10-11: Color count accumulators
        let (secret_counts, guess_counts) = calculate_color_counts(secret, guess);

        // Use row index to determine which color counts to store
        let color_idx = row % 6;
        columns[cols::AUX_S_COUNT].push(FieldElement::from(secret_counts[color_idx] as u64));
        columns[cols::AUX_G_COUNT].push(FieldElement::from(guess_counts[color_idx] as u64));
    }

    TraceTable::from_columns_main(columns, 1)
}

/// Calculate color counts for partial match calculation
fn calculate_color_counts(secret: &SecretCode, guess: &Guess) -> ([u8; 6], [u8; 6]) {
    let mut secret_counts = [0u8; 6];
    let mut guess_counts = [0u8; 6];

    for i in 0..4 {
        secret_counts[secret.0[i] as usize] += 1;
        guess_counts[guess.0[i] as usize] += 1;
    }

    (secret_counts, guess_counts)
}

/// Generate a trace table for proving knowledge of a valid secret
/// This version includes the actual computation of exact and partial matches
/// to prove they match the claimed feedback
pub fn generate_computation_trace<F: IsFFTField>(
    secret: &SecretCode,
    guess: &Guess,
) -> TraceTable<F, F> {
    let trace_length = 16;
    let num_cols = 12;

    // Create columns as vectors
    let mut columns: Vec<Vec<FieldElement<F>>> = vec![vec![]; num_cols];

    // Calculate the actual feedback
    let exact_count = count_exact_matches(secret, guess);
    let partial_count = count_partial_matches(secret, guess);

    // Calculate color counts
    let (secret_counts, guess_counts) = calculate_color_counts(secret, guess);

    // Convert to u64 values
    let secret_values: Vec<u64> = secret.0.iter().map(|c| *c as u64).collect();
    let guess_values: Vec<u64> = guess.0.iter().map(|c| *c as u64).collect();

    // Fill the trace table
    for row in 0..trace_length {
        // Secret columns
        for i in 0..4 {
            columns[cols::SECRET_0 + i].push(FieldElement::from(secret_values[i]));
        }

        // Guess columns
        for i in 0..4 {
            columns[cols::GUESS_0 + i].push(FieldElement::from(guess_values[i]));
        }

        // Auxiliary columns: store accumulated counts and intermediate values
        // Row 0: Store exact count and partial count
        if row == 0 {
            columns[cols::AUX_EXACT].push(FieldElement::from(exact_count as u64));
            columns[cols::AUX_PARTIAL].push(FieldElement::from(partial_count as u64));
        } else {
            // Other rows: store individual equality checks for verification
            let eq_idx = (row - 1) % 4;
            let s = secret_values[eq_idx];
            let g = guess_values[eq_idx];
            let is_eq = if s == g { 1 } else { 0 };
            columns[cols::AUX_EXACT].push(FieldElement::from(is_eq));
            columns[cols::AUX_PARTIAL].push(FieldElement::zero());
        }

        // Color count columns
        let color_idx = row % 6;
        columns[cols::AUX_S_COUNT].push(FieldElement::from(secret_counts[color_idx] as u64));
        columns[cols::AUX_G_COUNT].push(FieldElement::from(guess_counts[color_idx] as u64));
    }

    TraceTable::from_columns_main(columns, 1)
}

/// Count exact matches (color and position)
fn count_exact_matches(secret: &SecretCode, guess: &Guess) -> u8 {
    let mut count = 0;
    for i in 0..4 {
        if secret.0[i] == guess.0[i] {
            count += 1;
        }
    }
    count
}

/// Count partial matches (color only, not position)
fn count_partial_matches(secret: &SecretCode, guess: &Guess) -> u8 {
    let mut secret_counts = [0u8; 6];
    let mut guess_counts = [0u8; 6];
    let mut exact = 0;

    // Count exact matches and color frequencies
    for i in 0..4 {
        if secret.0[i] == guess.0[i] {
            exact += 1;
        }
        secret_counts[secret.0[i] as usize] += 1;
        guess_counts[guess.0[i] as usize] += 1;
    }

    // Calculate partial matches
    let mut partial = 0;
    for color in 0..6 {
        partial += std::cmp::min(secret_counts[color], guess_counts[color]);
    }
    partial -= exact;

    partial
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Color;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    type F = Stark252PrimeField;

    #[test]
    fn test_generate_trace() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let feedback = Feedback::new(2, 2);

        let trace = generate_trace::<F>(&secret, &guess, &feedback);

        assert_eq!(trace.num_rows(), 16);
        assert_eq!(trace.num_cols(), 12);

        // Check that secret is in column 0
        let s0 = &trace.main_table.get_row(0)[cols::SECRET_0];
        assert_eq!(*s0, FieldElement::from(0u64)); // Red = 0
    }

    #[test]
    fn test_color_counts() {
        let secret = SecretCode::new([Color::Red, Color::Red, Color::Blue, Color::Green]);
        let guess = Guess::new([Color::Red, Color::Blue, Color::Blue, Color::Yellow]);

        let (s_counts, g_counts) = calculate_color_counts(&secret, &guess);

        // Secret: 2 Red, 1 Blue, 1 Green
        assert_eq!(s_counts[0], 2); // Red
        assert_eq!(s_counts[1], 1); // Blue
        assert_eq!(s_counts[2], 1); // Green

        // Guess: 1 Red, 2 Blue, 1 Yellow
        assert_eq!(g_counts[0], 1); // Red
        assert_eq!(g_counts[1], 2); // Blue
        assert_eq!(g_counts[3], 1); // Yellow
    }
}
