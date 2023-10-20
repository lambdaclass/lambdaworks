use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::term::Term;

/// Wrapper struct for (coeff: FieldElement<F>, terms: Vec<usize>) representing a multivariate monomial in a sparse format.
// This sparse form is inspired by https://doc.sagemath.org/html/en/reference/polynomial_rings/sage/rings/polynomial/polydict.html
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiVariateMonomial<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub coeff: FieldElement<F>,
    pub vars: Vec<(usize, usize)>,
}

impl<F: IsField + IsPrimeField> MultiVariateMonomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Create a new `Term` from a tuple of the form `(coeff, (power))`
    fn new(term: (FieldElement<F>, Vec<(usize, usize)>)) -> Self {
        //todo: Check
        MultiVariateMonomial {
            coeff: term.0,
            vars: term.1,
        }
    }
}

impl<F: IsField + IsPrimeField> Term<F> for MultiVariateMonomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Returns the total degree of `self`. This is the sum of all variable
    /// powers in `self`
    fn degree(&self) -> usize {
        // A term in a multivariate monomial is distinguished by two
        self.vars.iter().fold(0, |acc, (_, y)| acc + y)
    }

    /// Returns a list of the powers of each variable in `self` i.e. numbers representing the id of the specific variable
    fn vars(&self) -> Vec<usize> {
        self.vars.iter().map(|(x, _)| *x).collect()
    }

    fn powers(&self) -> Vec<usize> {
        self.vars.iter().map(|(_, y)| *y).collect()
    }

    /// Fetches the max variable by id from the sparse list of id's this is used to ensure the upon evaluation the correct number of points are supplied
    // Sparse variables are stored in increasing var_id therefore we grab the last one
    fn max_var(&self) -> usize {
        self.vars.last().unwrap().0
    }

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        // check the number of evaluations points is equal to the number of variables
        assert_eq!(self.max_var(), p.len());
        // var_id is index of p
        let eval = self
            .vars
            .iter()
            .fold(FieldElement::<F>::one(), |acc, (x, y)| {
                acc * p[*x].pow::<usize>(*y)
            });
        eval * &self.coeff
    }

    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        todo!()
    }
}
