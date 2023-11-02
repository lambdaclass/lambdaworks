use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("xs and ys must be the same length. Got: {0} != {1}")]
    UnequalLengths(usize, usize),
    #[error("xs values should be unique.")]
    NonUniqueXs,
}

/// Trait defining the operations of Interpolating/Extending/Evaluting a polynomial of evaluations over a domain.
/// Coeffs of this polynomial are treated as evaluations over the domain.
pub trait IsEvaluation<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn from_evaluations(evaluations: &[F]) -> Self;

    //Name needs work -> think it should be type called multilinear extension
    fn evaluate_mle(&self) -> Self;

    //Name needs work -> think it should be type called multilinear extension
    fn evaluate_mle_with() -> Self;

    fn evaluate_vanishing() -> Self;

    fn evaluate_vanishing_with() -> Self;
    /// Returns a polynomial that interpolates the points with x coordinates and y coordinates given by
    /// `xs` and `ys`.
    /// `xs` and `ys` must be the same length, and `xs` values should be unique. If not, panics.
    fn interpolate(
        xs: &[FieldElement<F>],
        ys: &[FieldElement<F>],
    ) -> Result<Self, InterpolateError>
    where
        Self: Sized;

    fn vanishing(&self) -> Self;
}
