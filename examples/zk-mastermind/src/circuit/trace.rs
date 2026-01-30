//! Trace table generation for ZK Mastermind
//!
//! The trace table layout (14 columns):
//! - Columns 0-3: Secret code (private witness)
//! - Columns 4-7: Guess code (public input)
//! - Columns 8-9: Feedback (exact and partial counts)
//! - Columns 10-13: Equality indicators (eq_i = 1 if secret[i] == guess[i])

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
    pub const EQ_0: usize = 10;
    pub const EQ_1: usize = 11;
    pub const EQ_2: usize = 12;
    pub const EQ_3: usize = 13;
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
    let num_cols = 14;

    // Create columns as vectors of field elements
    let mut columns: Vec<Vec<FieldElement<F>>> = vec![vec![]; num_cols];

    // Convert secret and guess to u64 values
    let secret_values: Vec<u64> = secret.0.iter().map(|c| *c as u64).collect();
    let guess_values: Vec<u64> = guess.0.iter().map(|c| *c as u64).collect();

    // Compute equality indicators: eq_i = 1 if secret[i] == guess[i], else 0
    let eq_indicators: [u64; 4] = [
        if secret_values[0] == guess_values[0] { 1 } else { 0 },
        if secret_values[1] == guess_values[1] { 1 } else { 0 },
        if secret_values[2] == guess_values[2] { 1 } else { 0 },
        if secret_values[3] == guess_values[3] { 1 } else { 0 },
    ];

    // Fill the trace table
    for _row in 0..trace_length {
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

        // Columns 10-13: Equality indicators
        columns[cols::EQ_0].push(FieldElement::from(eq_indicators[0]));
        columns[cols::EQ_1].push(FieldElement::from(eq_indicators[1]));
        columns[cols::EQ_2].push(FieldElement::from(eq_indicators[2]));
        columns[cols::EQ_3].push(FieldElement::from(eq_indicators[3]));
    }

    TraceTable::from_columns_main(columns, 1)
}

/// Generate a trace table for proving knowledge of a valid secret
/// This version includes the actual computation of exact and partial matches
/// to prove they match the claimed feedback
pub fn generate_computation_trace<F: IsFFTField>(
    secret: &SecretCode,
    guess: &Guess,
) -> TraceTable<F, F> {
    let trace_length = 16;
    let num_cols = 14;

    // Create columns as vectors
    let mut columns: Vec<Vec<FieldElement<F>>> = vec![vec![]; num_cols];

    // Calculate the actual feedback
    let exact_count = count_exact_matches(secret, guess);
    let partial_count = count_partial_matches(secret, guess);

    // Convert to u64 values
    let secret_values: Vec<u64> = secret.0.iter().map(|c| *c as u64).collect();
    let guess_values: Vec<u64> = guess.0.iter().map(|c| *c as u64).collect();

    // Compute equality indicators: eq_i = 1 if secret[i] == guess[i], else 0
    let eq_indicators: [u64; 4] = [
        if secret_values[0] == guess_values[0] { 1 } else { 0 },
        if secret_values[1] == guess_values[1] { 1 } else { 0 },
        if secret_values[2] == guess_values[2] { 1 } else { 0 },
        if secret_values[3] == guess_values[3] { 1 } else { 0 },
    ];

    // Fill the trace table
    for _row in 0..trace_length {
        // Secret columns
        for i in 0..4 {
            columns[cols::SECRET_0 + i].push(FieldElement::from(secret_values[i]));
        }

        // Guess columns
        for i in 0..4 {
            columns[cols::GUESS_0 + i].push(FieldElement::from(guess_values[i]));
        }

        // Feedback columns (same in all rows)
        columns[cols::AUX_EXACT].push(FieldElement::from(exact_count as u64));
        columns[cols::AUX_PARTIAL].push(FieldElement::from(partial_count as u64));

        // Equality indicator columns (same in all rows)
        columns[cols::EQ_0].push(FieldElement::from(eq_indicators[0]));
        columns[cols::EQ_1].push(FieldElement::from(eq_indicators[1]));
        columns[cols::EQ_2].push(FieldElement::from(eq_indicators[2]));
        columns[cols::EQ_3].push(FieldElement::from(eq_indicators[3]));
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
        assert_eq!(trace.num_cols(), 14);

        // Check that secret is in column 0
        let s0 = &trace.main_table.get_row(0)[cols::SECRET_0];
        assert_eq!(*s0, FieldElement::from(0u64)); // Red = 0
    }

    #[test]
    fn test_equality_indicators() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);
        let feedback = Feedback::new(2, 2);

        let trace = generate_trace::<F>(&secret, &guess, &feedback);

        // Check equality indicators
        let row = trace.main_table.get_row(0);

        // Position 0: Red == Red -> eq_0 = 1
        assert_eq!(row[cols::EQ_0], FieldElement::from(1u64));
        // Position 1: Blue != Green -> eq_1 = 0
        assert_eq!(row[cols::EQ_1], FieldElement::from(0u64));
        // Position 2: Green != Blue -> eq_2 = 0
        assert_eq!(row[cols::EQ_2], FieldElement::from(0u64));
        // Position 3: Yellow == Yellow -> eq_3 = 1
        assert_eq!(row[cols::EQ_3], FieldElement::from(1u64));
    }

    #[test]
    fn test_computation_trace() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);

        let trace = generate_computation_trace::<F>(&secret, &guess);

        assert_eq!(trace.num_rows(), 16);
        assert_eq!(trace.num_cols(), 14);

        let row = trace.main_table.get_row(0);

        // Should have computed 2 exact matches (positions 0 and 3)
        assert_eq!(row[cols::AUX_EXACT], FieldElement::from(2u64));
        // Should have computed 2 partial matches (Blue and Green swapped)
        assert_eq!(row[cols::AUX_PARTIAL], FieldElement::from(2u64));
    }
}
