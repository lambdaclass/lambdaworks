use crate::{
    field::{
        element::FieldElement,
        traits::{IsField, IsPrimeField},
    },
    polynomial::traits::term::Term,
};
use core::fmt::Display;
use std::ops;

/// Wrapper struct for (coeff: FieldElement<F>, terms: Vec<usize>) representing a multivariate monomial in a sparse format.
// This sparse form is inspired by https://doc.sagemath.org/html/en/reference/polynomial_rings/sage/rings/polynomial/polydict.html
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MultivariateMonomial<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub coeff: FieldElement<F>,
    pub vars: Vec<(usize, usize)>,
}

impl<F: IsField + IsPrimeField> MultivariateMonomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Create a new `Term` from a tuple of the form `(coeff, (power))`
    pub fn new(term: (FieldElement<F>, Vec<(usize, usize)>)) -> Self {
        //todo: Check
        let mut vars = term.1;
        vars.sort();
        MultivariateMonomial {
            coeff: term.0,
            vars,
        }
    }
}

impl<F: IsField + IsPrimeField> Term<F> for MultivariateMonomial<F>
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
        match self.vars.last() {
            Some(&var) => var.0,
            None => 0,
        }
    }

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        // check the number of evaluations points is equal to the number of variables
        if self.vars.is_empty() {
            return self.coeff.clone();
        }

        // var_id is index of p
        let eval = self
            .vars
            .iter()
            .fold(FieldElement::<F>::one(), |acc, (x, y)| {
                acc * p[*x].pow::<usize>(*y)
            });
        eval * &self.coeff
    }

    //TODO: add valid variable check
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        let mut new_coefficient = self.coeff.clone();
        let mut unassigned_variables = self.vars.to_vec();
        let mut var_ids: Vec<usize> = unassigned_variables.iter().map(|x| x.0).collect();

        for (var_id, assignment) in assignments {
            if var_ids.contains(var_id) {
                new_coefficient = new_coefficient * assignment;
                unassigned_variables.retain(|&id| id.0 != *var_id);
                var_ids.retain(|&id| id != *var_id);
            }
        }

        Self::new((new_coefficient, unassigned_variables))
    }

    fn zero() -> Self {
        Self {
            coeff: FieldElement::<F>::zero(),
            vars: vec![],
        }
    }
}

// Add
impl<F: IsField + IsPrimeField> ops::Add<MultivariateMonomial<F>> for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn add(self, rhs: MultivariateMonomial<F>) -> Self::Output {
        Self {
            coeff: self.coeff + rhs.coeff,
            vars: self.vars,
        }
    }
}

impl<F: IsField + IsPrimeField> ops::AddAssign<MultivariateMonomial<F>> for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    fn add_assign(&mut self, rhs: MultivariateMonomial<F>) {
        self.coeff += rhs.coeff
    }
}

// Mul
impl<F: IsField + IsPrimeField> ops::Mul<MultivariateMonomial<F>> for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: MultivariateMonomial<F>) -> Self::Output {
        
        // check variables ids if match add power, else append a new variable, sort variables
        // TODO: what if they are different???
        let mut new_vars = self.vars.clone();
        for var in rhs.vars.clone().into_iter() {
            let mut added = false; // flag to check if the monomial was added or not

            //TODO: rayon
            //TODO: maybe check implementation of contains() to see if it only checks first item of tuple
            for term in new_vars.iter_mut() {
                if term.0 == var.0 {
                    term.1 += var.1;
                    added = true;
                }
            }

            if !added {
                new_vars.push(var);
            }
        }
        //TODO: check sorting
        new_vars.sort();
        Self {
            coeff: self.coeff * rhs.coeff,
            vars: new_vars
        } 
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&MultivariateMonomial<F>> for &MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: &MultivariateMonomial<F>) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<MultivariateMonomial<F>> for &MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: MultivariateMonomial<F>) -> Self::Output {
        self.clone() * rhs
    }
}

// Mul: Mono * Field

impl<F: IsField + IsPrimeField> ops::Mul<MultivariateMonomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: MultivariateMonomial<F>) -> Self::Output {
        MultivariateMonomial { coeff: rhs.coeff * self, vars: rhs.vars}
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&MultivariateMonomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: &MultivariateMonomial<F>) -> Self::Output {
        MultivariateMonomial { coeff: &rhs.coeff * self, vars: rhs.vars.clone()}
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<MultivariateMonomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: MultivariateMonomial<F>) -> Self::Output {
        MultivariateMonomial { coeff: rhs.coeff * self, vars: rhs.vars}
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&MultivariateMonomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: &MultivariateMonomial<F>) -> Self::Output {
        MultivariateMonomial { coeff: &rhs.coeff * self, vars: rhs.vars.clone()}
    }
}

// Mul: Field * Mono
impl<F: IsField + IsPrimeField> ops::Mul<FieldElement<F>> for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        MultivariateMonomial { coeff: self.coeff * rhs, vars: self.vars}
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&FieldElement<F>> for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        MultivariateMonomial { coeff: self.coeff * rhs, vars: self.vars}
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<FieldElement<F>> for &MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        MultivariateMonomial { coeff: &self.coeff * rhs, vars: self.vars.clone()}
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&FieldElement<F>> for &MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        MultivariateMonomial { coeff: &self.coeff * rhs, vars: self.vars.clone()}
    }
}

// Neg
impl<F: IsField + IsPrimeField> ops::Neg for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn neg(self) -> MultivariateMonomial<F> {
        MultivariateMonomial {
            coeff: -self.coeff,
            vars: self.vars,
        }
    }
}

impl<F: IsField + IsPrimeField> ops::Neg for &MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn neg(self) -> MultivariateMonomial<F> {
        MultivariateMonomial {
            coeff: -&self.coeff,
            vars: self.vars.clone(),
        }
    }
}

impl<F: IsField + IsPrimeField> Display for MultivariateMonomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.coeff.representative())
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
        let monomial = MultivariateMonomial::new((FE::new(5), vec![(10, 1), (5, 1), (6, 1)]));

        // should build and sort the var_id's
        assert_eq!(
            monomial,
            MultivariateMonomial {
                coeff: FE::new(5),
                vars: vec![(5, 1), (6, 1), (10, 1)]
            }
        );
    }

    #[test]
    fn evaluate_constant_multivariate_monomial() {
        let monomial = MultivariateMonomial::new((FE::new(5), vec![]));

        assert_eq!(monomial.evaluate(&[FE::new(1)]), FE::new(5),);
    }

    #[test]
    fn test_partial_evaluation() {
        // 5ab partially evaluate b = 2
        // expected result = 10a
        let five_a_b = MultivariateMonomial::new((FE::new(5), vec![(1, 1), (2, 1)]));
        let maybe_10_a = five_a_b.partial_evaluate(&[(2, FE::new(2))]);
        assert_eq!(
            maybe_10_a,
            MultivariateMonomial {
                coeff: FE::new(10),
                vars: vec![(1, 1)]
            }
        );

        // 6abcd evaluate a = 5, c = 3
        // expected = 90bd
        let six_a_b_c_d =
            MultivariateMonomial::new((FE::new(6), vec![(1, 1), (2, 1), (3, 1), (4, 1)]));
        let maybe_90_b_d = six_a_b_c_d.partial_evaluate(&[(1, FE::new(5)), (3, FE::new(3))]);
        assert_eq!(
            maybe_90_b_d,
            MultivariateMonomial {
                coeff: FE::new(90),
                vars: vec![(2, 1), (4, 1)]
            }
        );

        // assign every variable
        // 5ab partially evaluate a= 3, b = 2
        // expected result = 30
        let five_a_b = MultivariateMonomial::new((FE::new(5), vec![(1, 1), (2, 1)]));
        let maybe_30 = five_a_b.partial_evaluate(&[(1, FE::new(3)), (2, FE::new(2))]);
        assert_eq!(
            maybe_30,
            MultivariateMonomial {
                coeff: FE::new(30),
                vars: vec![]
            }
        );

        // ignore repeated assignments
        // 6abcd evaluate a = 5, c = 3, a = 9
        // expected = 90bd
        // should ignore the second assignment for a, as first already got rid of a
        let six_a_b_c_d =
            MultivariateMonomial::new((FE::new(6), vec![(1, 1), (2, 1), (3, 1), (4, 1)]));
        let maybe_90_b_d =
            six_a_b_c_d.partial_evaluate(&[(1, FE::new(5)), (3, FE::new(3)), (1, FE::new(9))]);
        assert_eq!(
            maybe_90_b_d,
            MultivariateMonomial {
                coeff: FE::new(90),
                vars: vec![(2, 1), (4, 1)]
            }
        );
    }
}
