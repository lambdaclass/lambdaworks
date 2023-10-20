use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::term::Term;

/// Struct for (coeff: FieldElement<F>, terms: Vec<usize>) representing a multilinear
/// monomial in a sparse format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiLinearMonomial<F: IsField + IsPrimeField>
    where
        <F as IsField>::BaseType: Send + Sync,
{
    pub coeff: FieldElement<F>,
    pub vars: Vec<usize>,
}

impl<F: IsField + IsPrimeField> MultiLinearMonomial<F>
    where
        <F as IsField>::BaseType: Send + Sync,
{
    /// Create a new `Term` from a tuple of the form `(coeff, (variables))`
    fn new(term: (FieldElement<F>, Vec<usize>)) -> Self {
        // sort variables in increasing order
        let mut vars = term.1;
        vars.sort();

        Self {
            coeff: term.0,
            vars,
        }
    }
}

impl<F: IsField + IsPrimeField> Term<F> for MultiLinearMonomial<F>
    where
        <F as IsField>::BaseType: Send + Sync,
{

    /// Returns the total degree of `self`. This is the count of all variables
    fn degree(&self) -> usize {
        self.vars.len()
    }

    /// Returns a list of the powers of each variable in `self` i.e. numbers representing the id of
    /// the specific variable
    fn vars(&self) -> Vec<usize> {
        self.vars.clone()
    }

    fn powers(&self) -> Vec<usize> {
        vec![1; self.vars.len()]
    }

    /// Fetches the max variable by id from the sparse list of id's this is used to ensure the upon
    /// evaluation the correct number of points are supplied
    fn max_var(&self) -> usize {
        // Sparse variables are stored in increasing var_id therefore we grab the last one
        *self.vars.last().unwrap()
    }

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        // check the number of evaluations points is equal to the number of variables
        assert_eq!(self.max_var(), p.len());
        // var_id is index of p
        let eval = self
            .vars
            .iter()
            .fold(FieldElement::<F>::one(), |acc, x| {
                acc * p[*x].clone()
            });
        eval * &self.coeff
    }

    // TODO: add documentation
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        let mut new_coefficient = self.coeff.clone();
        let mut unassigned_variables: Vec<usize> = vec![];

        // TODO: should not allow double assignments
        //  i.e repeated var_id
        for (var_id, assignment) in assignments {
            if self.vars.contains(var_id) {
                new_coefficient = new_coefficient * assignment;
            } else {
                unassigned_variables.push(*var_id);
            }
        }

        Self::new((new_coefficient, unassigned_variables))
    }
}

// TODO: add test to show that construction from non sorted vars work
// TODO: test partial evaluation
//        - also show that you cannot repeat var_id


#[cfg(test)]
mod tests {
    use crate::field::element::FieldElement;
    use crate::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::polynomial::multilinear_term::MultiLinearMonomial;

    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    #[test]
    fn build_multilinear_monomial() {
        let monomial = MultiLinearMonomial::new((FE::new(5), vec![10, 5, 6]));
    }
}