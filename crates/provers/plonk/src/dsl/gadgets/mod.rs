//! Composable circuit gadgets for common operations.
//!
//! Gadgets are reusable circuit components that encapsulate complex
//! constraint patterns. Each gadget implements the `Gadget` trait.
//!
//! # Available Gadgets
//!
//! - **Arithmetic**: Range checks, division with remainder
//! - **Comparison**: Less than, equality, zero check
//! - **Hash**: Poseidon, MiMC
//! - **Merkle**: Merkle proof verification

pub mod arithmetic;
pub mod comparison;
pub mod merkle;
pub mod poseidon;

use crate::dsl::builder::CircuitBuilder;
use lambdaworks_math::field::traits::IsField;

/// Error type for gadget synthesis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GadgetError {
    /// Invalid input to the gadget
    InvalidInput(String),
    /// Synthesis failed
    SynthesisError(String),
    /// Constraint violation
    ConstraintViolation(String),
}

impl std::fmt::Display for GadgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GadgetError::InvalidInput(msg) => write!(f, "Invalid gadget input: {}", msg),
            GadgetError::SynthesisError(msg) => write!(f, "Gadget synthesis error: {}", msg),
            GadgetError::ConstraintViolation(msg) => write!(f, "Constraint violation: {}", msg),
        }
    }
}

impl std::error::Error for GadgetError {}

/// A composable circuit component.
///
/// Gadgets encapsulate common circuit patterns and can be composed
/// to build complex circuits. Each gadget defines its input and output
/// types and how to synthesize constraints.
///
/// # Example Implementation
///
/// ```ignore
/// struct IsZeroGadget;
///
/// impl<F: IsField> Gadget<F> for IsZeroGadget {
///     type Input = FieldVar;
///     type Output = BoolVar;
///
///     fn synthesize(
///         builder: &mut CircuitBuilder<F>,
///         input: Self::Input,
///     ) -> Result<Self::Output, GadgetError> {
///         // ... implementation
///     }
///
///     fn constraint_count() -> usize { 2 }
///     fn name() -> &'static str { "IsZero" }
/// }
/// ```
pub trait Gadget<F: IsField> {
    /// The input type for this gadget.
    type Input;

    /// The output type for this gadget.
    type Output;

    /// Synthesizes the gadget's constraints.
    ///
    /// This method adds constraints to the circuit builder and returns
    /// output variables.
    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        input: Self::Input,
    ) -> Result<Self::Output, GadgetError>;

    /// Returns the number of constraints this gadget adds.
    ///
    /// This is useful for cost estimation and circuit size planning.
    fn constraint_count() -> usize;

    /// Returns a human-readable name for debugging.
    fn name() -> &'static str;
}
