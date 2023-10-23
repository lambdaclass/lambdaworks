use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use std::fmt::Debug;

/// Term represents an abstraction of individual monomial term of a polynomial.
/// Ex:
///     Unvariate: (coeff) -> FieldElement<F>
///     Multilinear: (coeff, Vec<var_id>) -> (FieldElement<F>, Vec<usize>)
///     MultiVariate: (coeff, Vec<(var_id, power)>) -> (FieldLElement<F>, Vec<(usize, usize))
// NOTE: open question whether we condense the Multilinear and Multivariate terms in to a single Multivariate term and iterator using .chunks()
pub trait Term<F: IsField>: Clone + Debug + Send + Sync {
    /// Returns the total degree of `self`. This is the sum of all variable
    /// powers in `self`
    fn degree(&self) -> usize;

    /// Returns a list of variables in `self` i.e. numbers representing the id of the specific variable 0: x0, 1: x1, 2: x2, etc.
    fn vars(&self) -> Vec<usize>;

    /// Returns a list of the powers of each variable in `self`
    fn powers(&self) -> Vec<usize>;

    /// Fetches the max variable by id from the sparse list of id's this is used to ensure the upon evaluation the correct number of points are supplied
    /// Note: returns 0 if constant term
    fn max_var(&self) -> usize;

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F>;

    // TODO: add documentation
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self;
}
