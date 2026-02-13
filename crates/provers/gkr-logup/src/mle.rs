use core::ops::Index;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Multilinear extension stored as evaluations on the boolean hypercube in lexicographic order.
///
/// Provides an in-place `fix_first_variable` that avoids allocating a new vector each round,
/// unlike `DenseMultilinearPolynomial`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mle<F: IsField> {
    evals: Vec<FieldElement<F>>,
}

impl<F: IsField> Mle<F> {
    /// Creates a new MLE from evaluations. Length must be a power of two.
    pub fn new(evals: Vec<FieldElement<F>>) -> Self {
        assert!(
            evals.len().is_power_of_two(),
            "MLE length must be a power of two, got {}",
            evals.len()
        );
        Self { evals }
    }

    /// Number of variables (log2 of length).
    pub fn n_variables(&self) -> usize {
        self.evals.len().trailing_zeros() as usize
    }

    /// Number of evaluations.
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    /// Whether the evaluation table is empty.
    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    /// Returns the evaluation at the given index.
    pub fn at(&self, index: usize) -> FieldElement<F> {
        self.evals[index].clone()
    }

    /// Returns a reference to the evaluations.
    pub fn evals(&self) -> &[FieldElement<F>] {
        &self.evals
    }

    /// Consumes self and returns the evaluations vector.
    pub fn into_evals(self) -> Vec<FieldElement<F>> {
        self.evals
    }

    /// Fixes the first variable to `r` **in-place**, halving the number of evaluations.
    ///
    /// For lexicographic order: the first half has x_0=0, the second half has x_0=1.
    /// `evals[j] = evals[j] + r * (evals[j + half] - evals[j])` for j in 0..half,
    /// then truncate to `half`.
    pub fn fix_first_variable(&mut self, r: &FieldElement<F>) {
        let n = self.n_variables();
        assert!(n > 0, "Cannot fix variable in a 0-variable MLE");
        let half = self.evals.len() / 2;
        for j in 0..half {
            let diff = self.evals[j + half].clone() - &self.evals[j];
            self.evals[j] = &self.evals[j] + &(r * diff);
        }
        self.evals.truncate(half);
    }
}

impl<F: IsField> Index<usize> for Mle<F> {
    type Output = FieldElement<F>;

    #[inline(always)]
    fn index(&self, index: usize) -> &FieldElement<F> {
        &self.evals[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn basic_properties() {
        let mle = Mle::new(vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)]);
        assert_eq!(mle.n_variables(), 2);
        assert_eq!(mle.len(), 4);
        assert_eq!(mle.at(0), FE::from(1));
        assert_eq!(mle[3], FE::from(4));
    }

    #[test]
    fn fix_first_variable_matches_dense() {
        let evals = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let r = FE::from(7);

        // Our in-place version
        let mut mle = Mle::new(evals.clone());
        mle.fix_first_variable(&r);

        // Reference: DenseMultilinearPolynomial
        let dense = DenseMultilinearPolynomial::new(evals);
        let dense_fixed = dense.fix_first_variable(&r);

        assert_eq!(mle.evals(), dense_fixed.evals());
    }

    #[test]
    fn fix_first_variable_3vars_matches_dense() {
        let evals: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let r = FE::from(13);

        let mut mle = Mle::new(evals.clone());
        mle.fix_first_variable(&r);

        let dense = DenseMultilinearPolynomial::new(evals);
        let dense_fixed = dense.fix_first_variable(&r);

        assert_eq!(mle.evals(), dense_fixed.evals());
    }

    #[test]
    fn fix_all_variables_gives_single_value() {
        let evals: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let challenges = [FE::from(3), FE::from(7), FE::from(11)];

        let mut mle = Mle::new(evals.clone());
        for r in &challenges {
            mle.fix_first_variable(r);
        }

        let dense = DenseMultilinearPolynomial::new(evals);
        let expected = dense.evaluate(challenges.to_vec()).unwrap();

        assert_eq!(mle.len(), 1);
        assert_eq!(mle[0], expected);
    }
}
