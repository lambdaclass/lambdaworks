//! Game rules for Mastermind
//!
//! Calculates feedback (exact and partial matches) for a guess against a secret code.

use super::types::{Feedback, Guess, SecretCode};

/// Calculate the feedback for a guess against a secret code
///
/// # Arguments
/// * `secret` - The secret code (4 colors)
/// * `guess` - The guess (4 colors)
///
/// # Returns
/// * `Feedback` - The number of exact matches (correct color in correct position)
///   and partial matches (correct color in wrong position)
///
/// # Algorithm
/// 1. Count exact matches first (same position, same color)
/// 2. Count color frequencies for both secret and guess
/// 3. Partial matches = sum of min frequencies - exact matches
pub fn calculate_feedback(secret: &SecretCode, guess: &Guess) -> Feedback {
    let mut exact = 0u8;
    let mut secret_counts = [0u8; 6]; // One count per color
    let mut guess_counts = [0u8; 6];

    // First pass: count exact matches and color frequencies
    for i in 0..4 {
        let s = secret.0[i] as usize;
        let g = guess.0[i] as usize;

        if s == g {
            exact += 1;
        }

        secret_counts[s] += 1;
        guess_counts[g] += 1;
    }

    // Second pass: calculate partial matches
    // For each color, the number of partial matches is the minimum count
    // between secret and guess, minus any exact matches already counted
    let mut partial = 0u8;
    for color in 0..6 {
        partial += std::cmp::min(secret_counts[color], guess_counts[color]);
    }

    // Subtract exact matches since they were included in the min counts
    partial -= exact;

    Feedback { exact, partial }
}

/// Verify that calculated feedback matches the expected feedback
pub fn verify_feedback(secret: &SecretCode, guess: &Guess, expected: &Feedback) -> bool {
    let calculated = calculate_feedback(secret, guess);
    calculated == *expected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Color;

    #[test]
    fn test_all_exact() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let feedback = calculate_feedback(&secret, &guess);
        assert_eq!(feedback.exact, 4);
        assert_eq!(feedback.partial, 0);
    }

    #[test]
    fn test_no_matches() {
        let secret = SecretCode::new([Color::Red, Color::Red, Color::Red, Color::Red]);
        let guess = Guess::new([Color::Blue, Color::Blue, Color::Blue, Color::Blue]);
        let feedback = calculate_feedback(&secret, &guess);
        assert_eq!(feedback.exact, 0);
        assert_eq!(feedback.partial, 0);
    }

    #[test]
    fn test_all_partial() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Yellow, Color::Green, Color::Blue, Color::Red]);
        let feedback = calculate_feedback(&secret, &guess);
        assert_eq!(feedback.exact, 0);
        assert_eq!(feedback.partial, 4);
    }

    #[test]
    fn test_mixed() {
        // Secret: [R, B, G, Y]
        // Guess:  [R, G, B, O]
        // Exact: R at position 0
        // Partial: G and B swapped
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Orange]);
        let feedback = calculate_feedback(&secret, &guess);
        assert_eq!(feedback.exact, 1);
        assert_eq!(feedback.partial, 2);
    }

    #[test]
    fn test_duplicate_colors() {
        // Secret: [R, R, B, G]
        // Guess:  [R, R, R, B]
        // Exact: Two R's at positions 0,1
        // Partial: One B at position 2 (guess has R)
        let secret = SecretCode::new([Color::Red, Color::Red, Color::Blue, Color::Green]);
        let guess = Guess::new([Color::Red, Color::Red, Color::Red, Color::Blue]);
        let feedback = calculate_feedback(&secret, &guess);
        assert_eq!(feedback.exact, 2);
        assert_eq!(feedback.partial, 1);
    }

    #[test]
    fn test_duplicate_colors_partial_only() {
        // Secret: [R, R, B, G]
        // Guess:  [R, B, R, B]
        // Exact: One R at position 0
        // Partial: One B at position 1, One R at position 2
        let secret = SecretCode::new([Color::Red, Color::Red, Color::Blue, Color::Green]);
        let guess = Guess::new([Color::Red, Color::Blue, Color::Red, Color::Blue]);
        let feedback = calculate_feedback(&secret, &guess);
        // Secret counts: R=2, B=1, G=1
        // Guess counts: R=2, B=2
        // Exact: position 0 (R matches)
        // Partial: min(2,2)-1 for R (1), min(1,2) for B (1) = 2
        assert_eq!(feedback.exact, 1);
        assert_eq!(feedback.partial, 2);
    }

    #[test]
    fn test_verify_feedback() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let guess = Guess::new([Color::Red, Color::Green, Color::Blue, Color::Yellow]);

        // Expected: 2 exact (R, Y), 2 partial (B, G swapped)
        let correct = Feedback::new(2, 2);
        let incorrect = Feedback::new(4, 0);

        assert!(verify_feedback(&secret, &guess, &correct));
        assert!(!verify_feedback(&secret, &guess, &incorrect));
    }
}
