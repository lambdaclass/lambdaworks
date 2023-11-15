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
    // TODO: Consider making this an option
    fn vars(&self) -> Vec<usize>;

    /// Returns a list of the powers of each variable in `self`
    // TODO: Consider making this an option
    fn powers(&self) -> Vec<usize>;

    /// Fetches the max variable by id from the sparse list of id's this is used to ensure the upon evaluation the correct number of points are supplied
    /// Note: returns 0 if constant term
    fn max_var(&self) -> usize;

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F>;

    // TODO: add documentation
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self;

    fn zero() -> Self;
}

impl<F: IsField> Term<F> for FieldElement<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Returns the total degree of `self`. This is the sum of all variable
    /// powers in `self`
    fn degree(&self) -> usize {
        1
    }

    /// Returns a list of variables in `self` i.e. numbers representing the id of the specific variable 0: x0, 1: x1, 2: x2, etc.
    fn vars(&self) -> Vec<usize> {
        vec![]
    }

    /// Returns a list of the powers of each variable in `self`
    fn powers(&self) -> Vec<usize> {
        vec![]
    }

    /// Fetches the max variable by id from the sparse list of id's this is used to ensure the upon evaluation the correct number of points are supplied
    /// Note: returns 0 if constant term or univariate
    fn max_var(&self) -> usize {
        0
    }

    /// Evaluates `self` at the point `p`.
    // NOTE: In Univariate case p will be raised to the power^index therefore to evaluate we multiply
    // TODO: make result add check
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        self * &p[0]
    }

    // TODO: add documentation
    // TODO: Decide if this should be abstracted better?
    // TODO: make result add check
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        self * &assignments[0].1
    }

    fn zero() -> Self {
        FieldElement::zero()
    }
}
