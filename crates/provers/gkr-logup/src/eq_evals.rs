use core::ops::Index;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Precomputed evaluations of `eq((0, x), y)` for all `x` in `{0, 1}^(n-1)`.
///
/// Stored in lexicographic order. These evaluations allow the prover to compute
/// sums involving `eq(x, y)` efficiently in a single pass.
#[derive(Debug, Clone)]
pub struct EqEvaluations<F: IsField> {
    y: Vec<FieldElement<F>>,
    evals: Vec<FieldElement<F>>,
}

impl<F: IsField> EqEvaluations<F> {
    /// Generates all evaluations of `eq((0, x), y)` for `x` in `{0, 1}^(n-1)`.
    ///
    /// Uses the recursive structure of eq:
    /// `eq(x, y) = prod_i (x_i * y_i + (1 - x_i) * (1 - y_i))`
    ///
    /// The generation starts with `v = eq(0, y[0])` and doubles the array
    /// at each step using the next `y_i`.
    pub fn generate(y: &[FieldElement<F>]) -> Self {
        let y_vec = y.to_vec();

        if y.is_empty() {
            return Self {
                y: y_vec,
                evals: vec![FieldElement::one()],
            };
        }

        // Start with v = eq(0, y[0]) = 1 - y[0]
        let v = FieldElement::<F>::one() - &y[0];
        let evals = gen_eq_evals(&y[1..], v);

        debug_assert_eq!(evals.len(), 1 << (y.len() - 1));

        Self { y: y_vec, evals }
    }

    /// Returns the fixed vector `y` used to generate the evaluations.
    pub fn y(&self) -> &[FieldElement<F>] {
        &self.y
    }

    /// Returns a reference to the evaluations.
    pub fn evals(&self) -> &[FieldElement<F>] {
        &self.evals
    }
}

impl<F: IsField> Index<usize> for EqEvaluations<F> {
    type Output = FieldElement<F>;

    #[inline(always)]
    fn index(&self, index: usize) -> &FieldElement<F> {
        &self.evals[index]
    }
}

/// Returns evaluations `eq(x, y) * v` for all `x` in `{0, 1}^n` in lexicographic order.
///
/// Algorithm: start with `evals = [v]`, then for each `y_i` (processed in reverse
/// for lexicographic layout), double the array:
/// - `evals[j]` becomes `evals[j] * (1 - y_i)` (the x_i=0 case)
/// - new entry `evals[j] * y_i` is appended (the x_i=1 case)
///
/// Processing `y` in reverse order produces lexicographic output, matching
/// lambdaworks' MLE convention.
pub fn gen_eq_evals<F: IsField>(y: &[FieldElement<F>], v: FieldElement<F>) -> Vec<FieldElement<F>> {
    let mut evals = Vec::with_capacity(1 << y.len());
    evals.push(v);

    for y_i in y.iter().rev() {
        let len = evals.len();
        for j in 0..len {
            // tmp = evals[j] * y_i (the x_i=1 branch)
            let tmp = &evals[j] * y_i;
            // evals[j] = evals[j] * (1 - y_i) = evals[j] - tmp
            evals.push(tmp.clone());
            evals[j] = &evals[j] - &tmp;
        }
    }

    evals
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::eq_eval;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn gen_eq_evals_matches_eq_eval() {
        let y = vec![FE::from(7), FE::from(3)];
        let v = FE::from(2);
        let evals = gen_eq_evals(&y, v);

        let zero = FE::zero();
        let one = FE::one();

        // Check eq(x, y) * v for all x in {0,1}^2 in lexicographic order
        assert_eq!(evals[0], eq_eval(&[zero, zero], &y) * v); // x=(0,0)
        assert_eq!(evals[1], eq_eval(&[zero, one], &y) * v); // x=(0,1)
        assert_eq!(evals[2], eq_eval(&[one, zero], &y) * v); // x=(1,0)
        assert_eq!(evals[3], eq_eval(&[one, one], &y) * v); // x=(1,1)
    }

    #[test]
    fn eq_evals_generate_matches_eq_eval() {
        let y = vec![FE::from(5), FE::from(11), FE::from(13)];
        let eq_e = EqEvaluations::generate(&y);

        let zero = FE::zero();
        let one = FE::one();

        // EqEvaluations stores eq((0, x), y) for x in {0,1}^2
        // eq((0, x1, x2), (y0, y1, y2)) = eq(0, y0) * eq(x1, y1) * eq(x2, y2)
        assert_eq!(eq_e.evals().len(), 4);

        for (idx, (x1, x2)) in [(zero, zero), (zero, one), (one, zero), (one, one)]
            .iter()
            .enumerate()
        {
            let expected = eq_eval(&[zero, *x1, *x2], &y);
            assert_eq!(eq_e[idx], expected, "mismatch at index {idx}");
        }
    }

    #[test]
    fn eq_evals_empty_y() {
        let eq_e = EqEvaluations::<F>::generate(&[]);
        assert_eq!(eq_e.evals().len(), 1);
        assert_eq!(eq_e[0], FE::one());
    }

    #[test]
    fn eq_evals_single_y() {
        let y = vec![FE::from(7)];
        let eq_e = EqEvaluations::generate(&y);
        // eq((0,), (7,)) = (1 - 0)*(1 - 7) + 0*7 = 1 - 7 = -6 = 95
        assert_eq!(eq_e.evals().len(), 1);
        assert_eq!(eq_e[0], FE::one() - FE::from(7));
    }
}
