use crate::errors::TermError;
use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use crate::polynomial::traits::term::Term;

//TODO: generalize Multivarite division and add it
pub trait IsPolynomial<F: IsField, T: Term<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type T;

    /// Build a new multilinear polynomial, from collection of multilinear monomials
    fn new(num_vars: usize, terms: &[T]) -> Self;

    //Returns the degree of the polynomial
    fn degree(&self) -> usize;

    fn leading_coefficient(&self) -> FieldElement<F>;

    fn coeffs(&self) -> &[FieldElement<F>];

    fn len(&self) -> usize;

    /// Adds an additional term to the monomial
    fn extend(&mut self, term: T);

    /// Evaluates `self` at the point `p`.
    /// Note: assumes p contains points for all variables aka is not sparse.
    fn evaluate(&self, p: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError>;

    fn evaluate_with(terms: &[T], p: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError>;

    /// Selectively assign values to variables in the polynomial, returns a reduced
    /// polynomial after assignment evaluation
    // TODO: can we change this to modify in place to remove the extract allocation
    // TODO: maybe change to fixed?

    fn eval_at_one(&self) -> FieldElement<F>;

    fn eval_at_zero(&self) -> FieldElement<F>;

    fn zero() -> Self;

    //Two cases is_empty || coeffs are all zeros
    fn is_zero(&self) -> bool;

    // Supports abstraction as a vector
    fn is_empty(&self) -> bool;

    fn pad_with_zero_coefficients_to_length(pa: &mut Self, n: usize);

    fn pad_with_zero_coefficients(pa: &Self, pb: &Self) -> (Self, Self)
    where
        Self: Sized;

    fn scale(&self, factor: &FieldElement<F>) -> Self;

    fn scale_coeffs(&self, factor: &FieldElement<F>) -> Self;

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    fn split_n_ways(&self, n: usize) -> Vec<Self>
    where
        Self: Sized;
}
