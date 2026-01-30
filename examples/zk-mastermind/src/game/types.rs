//! Game logic types for ZK Mastermind
//!
//! Mastermind uses 6 colors and a secret code of 4 positions.

use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    traits::IsFFTField,
};

/// Felt252 type alias for convenience
pub type Felt252 = FieldElement<Stark252PrimeField>;

/// The 6 colors in Mastermind, encoded as field elements 0-5
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    Red = 0,
    Blue = 1,
    Green = 2,
    Yellow = 3,
    Orange = 4,
    Purple = 5,
}

impl Color {
    /// Convert Color to field element
    pub fn to_field(&self) -> Felt252 {
        Felt252::from(*self as u64)
    }

    /// Create Color from u64 value (must be 0-5)
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Color::Red),
            1 => Some(Color::Blue),
            2 => Some(Color::Green),
            3 => Some(Color::Yellow),
            4 => Some(Color::Orange),
            5 => Some(Color::Purple),
            _ => None,
        }
    }

    /// Get all colors as an array
    pub fn all() -> [Color; 6] {
        [
            Color::Red,
            Color::Blue,
            Color::Green,
            Color::Yellow,
            Color::Orange,
            Color::Purple,
        ]
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Color::Red => write!(f, "R"),
            Color::Blue => write!(f, "B"),
            Color::Green => write!(f, "G"),
            Color::Yellow => write!(f, "Y"),
            Color::Orange => write!(f, "O"),
            Color::Purple => write!(f, "P"),
        }
    }
}

/// The secret code: 4 colors chosen by the CodeMaker
#[derive(Clone, Debug, PartialEq)]
pub struct SecretCode(pub [Color; 4]);

impl SecretCode {
    /// Create a new secret code
    pub fn new(colors: [Color; 4]) -> Self {
        Self(colors)
    }

    /// Convert to field elements
    pub fn to_fields(&self) -> [Felt252; 4] {
        [
            self.0[0].to_field(),
            self.0[1].to_field(),
            self.0[2].to_field(),
            self.0[3].to_field(),
        ]
    }
}

impl std::fmt::Display for SecretCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

/// A guess: 4 colors chosen by the CodeBreaker
#[derive(Clone, Debug, PartialEq)]
pub struct Guess(pub [Color; 4]);

impl Guess {
    /// Create a new guess
    pub fn new(colors: [Color; 4]) -> Self {
        Self(colors)
    }

    /// Convert to field elements
    pub fn to_fields(&self) -> [Felt252; 4] {
        [
            self.0[0].to_field(),
            self.0[1].to_field(),
            self.0[2].to_field(),
            self.0[3].to_field(),
        ]
    }
}

impl std::fmt::Display for Guess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

/// Feedback for a guess: exact matches (color + position) and partial matches (color only)
#[derive(Clone, Debug, PartialEq)]
pub struct Feedback {
    /// Number of exact matches (correct color in correct position)
    pub exact: u8,
    /// Number of partial matches (correct color in wrong position)
    pub partial: u8,
}

impl Feedback {
    /// Create new feedback
    pub fn new(exact: u8, partial: u8) -> Self {
        Self { exact, partial }
    }

    /// Check if the guess completely solved the code
    pub fn is_win(&self) -> bool {
        self.exact == 4
    }

    /// Convert to field elements for the circuit
    pub fn to_fields(&self) -> [Felt252; 2] {
        [
            Felt252::from(self.exact as u64),
            Felt252::from(self.partial as u64),
        ]
    }
}

impl std::fmt::Display for Feedback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} exact, {} partial", self.exact, self.partial)
    }
}

/// Public inputs for the ZK proof (generic version)
/// F is the field type (e.g., Stark252PrimeField), not FieldElement<F>
#[derive(Clone, Debug)]
pub struct MastermindPublicInputs<F: IsFFTField> {
    /// The guess made by the CodeBreaker
    pub guess: [FieldElement<F>; 4],
    /// The expected feedback
    pub feedback: [FieldElement<F>; 2],
}

impl MastermindPublicInputs<Stark252PrimeField> {
    pub fn new(guess: &Guess, feedback: &Feedback) -> Self {
        Self {
            guess: guess.to_fields(),
            feedback: feedback.to_fields(),
        }
    }
}

impl<F: IsFFTField> MastermindPublicInputs<F> {
    /// Create from raw field elements
    pub fn from_fields(guess: [FieldElement<F>; 4], feedback: [FieldElement<F>; 2]) -> Self {
        Self { guess, feedback }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_to_field() {
        assert_eq!(Color::Red.to_field(), Felt252::from(0));
        assert_eq!(Color::Blue.to_field(), Felt252::from(1));
        assert_eq!(Color::Purple.to_field(), Felt252::from(5));
    }

    #[test]
    fn test_color_from_u64() {
        assert_eq!(Color::from_u64(0), Some(Color::Red));
        assert_eq!(Color::from_u64(5), Some(Color::Purple));
        assert_eq!(Color::from_u64(6), None);
    }

    #[test]
    fn test_secret_code_to_fields() {
        let secret = SecretCode::new([Color::Red, Color::Blue, Color::Green, Color::Yellow]);
        let fields = secret.to_fields();
        assert_eq!(fields[0], Felt252::from(0));
        assert_eq!(fields[1], Felt252::from(1));
        assert_eq!(fields[2], Felt252::from(2));
        assert_eq!(fields[3], Felt252::from(3));
    }

    #[test]
    fn test_feedback_is_win() {
        assert!(Feedback::new(4, 0).is_win());
        assert!(!Feedback::new(3, 1).is_win());
        assert!(!Feedback::new(0, 4).is_win());
    }
}
