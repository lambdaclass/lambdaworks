//Dense and univariate term -> single coefficient
use crate::{
    field::{element::FieldElement, traits::IsField},
    polynomial::traits::term::Term
};

/// Wrapper struct for (coeff: FieldElement<F>, terms: Vec<usize>) representing a multivariate monomial in a sparse format.
// This sparse form is inspired by https://doc.sagemath.org/html/en/reference/polynomial_rings/sage/rings/polynomial/polydict.html
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DenseMonomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub coeff: FieldElement<F>,
    pub power: usize,
}

impl<F: IsField> DenseMonomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Create a new `Term` from a tuple of the form `(coeff, (power))`
    /// NOTE: for the Univariate case an abstraction like this doesn't make to much sense
    pub fn new(coeff: FieldElement<F>, power: usize) -> Self {
        //todo: Check
        DenseMonomial {
            coeff,
            power,
        }
    }
}

impl<F: IsField> Term<F> for DenseMonomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Returns the total degree of `self`. This is the sum of all variable
    /// powers in `self`
    fn degree(&self) -> usize {
        self.power
    } 

    /// Returns a list of the powers of each variable in `self` i.e. numbers representing the id of the specific variable
    /// TODO: maybe make this an option??? -> single variable has var_id == 0.
    fn vars(&self) -> Vec<usize> {
        vec![0]
    }

    fn powers(&self) -> Vec<usize> {
        vec![self.power]
    }

    /// Fetches the max variable by id from the sparse list of id's this is used to ensure the upon evaluation the correct number of points are supplied
    // Sparse variables are stored in increasing var_id therefore we grab the last one
    fn max_var(&self) -> usize {
        0
    }

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        // var_id is index of p
        &self.coeff * p[0].pow::<usize>(self.power)
    }

    //TODO: add valid variable check and see if this makes sense, probs doesn't lol
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {

        Self::new(&self.coeff * assignments[0].1.pow::<usize>(self.power), self.power)
    }
}

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::U64PrimeField;

    use super::*;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    #[test]
    fn build_multivariate_monomial() {
        let monomial = DenseMonomial::new(FE::new(5), 1);

        // should build and sort the var_id's
        assert_eq!(
            monomial,
            DenseMonomial {
                coeff: FE::new(5),
                power: 1
            }
        );
    }

    #[test]
    fn test_partial_evaluation() {
        // 5ab partially evaluate b = 2
        // expected result = 10a
        let five_a_b = DenseMonomial::new(FE::new(5), 1);
        let maybe_10_a = five_a_b.partial_evaluate(&[(2, FE::new(2))]);
        assert_eq!(
            maybe_10_a,
            DenseMonomial {
                coeff: FE::new(10),
                power: 1
            }
        );

        // 6abcd evaluate a = 5, c = 3
        // expected = 90bd
        let six_a_b_c_d =
            DenseMonomial::new(FE::new(6), 2);
        let maybe_90_b_d = six_a_b_c_d.partial_evaluate(&[(1, FE::new(5)), (3, FE::new(3))]);
        assert_eq!(
            maybe_90_b_d,
            DenseMonomial {
                coeff: FE::new(90),
                power: 1
            }
        );

        // assign every variable
        // 5ab partially evaluate a= 3, b = 2
        // expected result = 30
        let five_a_b = DenseMonomial::new(FE::new(5), 2);
        let maybe_30 = five_a_b.partial_evaluate(&[(1, FE::new(3)), (2, FE::new(2))]);
        assert_eq!(
            maybe_30,
            DenseMonomial {
                coeff: FE::new(30),
                power: 1
            }
        );

        // ignore repeated assignments
        // 6abcd evaluate a = 5, c = 3, a = 9
        // expected = 90bd
        // should ignore the second assignment for a, as first already got rid of a
        let six_a_b_c_d =
            DenseMonomial::new(FE::new(6), 3);
        let maybe_90_b_d =
            six_a_b_c_d.partial_evaluate(&[(1, FE::new(5)), (3, FE::new(3)), (1, FE::new(9))]);
        assert_eq!(
            maybe_90_b_d,
            DenseMonomial {
                coeff: FE::new(90),
                power: 1
            }
        );
    }
}
