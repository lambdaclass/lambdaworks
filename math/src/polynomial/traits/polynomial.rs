use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use crate::polynomial::traits::term::Term;
use crate::polynomial::InterpolateError;

pub trait IsPolynomial<F: IsField, T: Term<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type T;

    /// Build a new multilinear polynomial, from collection of multilinear monomials
    fn new(terms: &[T]) -> Self;

    /// Adds an additional term to the monomial
    fn extend(&mut self, term: T);

    /// Evaluates `self` at the point `p`.
    /// Note: assumes p contains points for all variables aka is not sparse.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F>;

    /// Selectively assign values to variables in the polynomial, returns a reduced
    /// polynomial after assignment evaluation
    // TODO: can we change this to modify in place to remove the extract allocation
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self;

    fn degree(&self) -> usize;

    fn leading_coefficient(&self) -> FieldElement<F>;

    fn coefficients(&self) -> &[FieldElement<F>];

    fn coeff_len(&self) -> usize;

    fn zero(&self) -> Self;

    fn vanishing(&self) -> Self;

    fn pad_with_zero_coefficients_to_length(pa: &mut Self, n: usize);

    fn pad_with_zero_coefficients(pa: &Self, pb: &Self) -> (Self, Self)
    where
        Self: Sized;

    /// Returns a polynomial that interpolates the points with x coordinates and y coordinates given by
    /// `xs` and `ys`.
    /// `xs` and `ys` must be the same length, and `xs` values should be unique. If not, panics.
    fn interpolate(
        xs: &[FieldElement<F>],
        ys: &[FieldElement<F>],
    ) -> Result<Self, InterpolateError>
    where
        Self: Sized;

    /// Computes quotient with `x - b` in place.
    fn ruffini_division_inplace(&mut self, b: &FieldElement<F>);

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: (quotient, remainder)
    fn long_division_with_remainder(self, dividend: &Self) -> (Self, Self)
    where
        Self: Sized;

    fn scale(&self, factor: &FieldElement<F>) -> Self;

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    fn break_in_parts(&self, number_of_parts: usize) -> Vec<Self>
    where
        Self: Sized;
}
