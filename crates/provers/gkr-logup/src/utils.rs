use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Evaluates `a_0 + a_1*x + ... + a_d*x^d` using Horner's method.
pub fn horner_eval<F: IsField>(coeffs: &[FieldElement<F>], x: &FieldElement<F>) -> FieldElement<F> {
    coeffs
        .iter()
        .rev()
        .fold(FieldElement::zero(), |acc, coeff| &acc * x + coeff)
}

/// Evaluates the eq polynomial: `eq(x, y) = prod_i (x_i * y_i + (1 - x_i) * (1 - y_i))`.
pub fn eq<F: IsField>(x: &[FieldElement<F>], y: &[FieldElement<F>]) -> FieldElement<F> {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let one = FieldElement::<F>::one();
            xi * yi + (&one - xi) * (&one - yi)
        })
        .fold(FieldElement::<F>::one(), |acc, term| acc * term)
}

/// Linearly interpolates between two MLE evaluations:
/// `fold_mle_evals(r, v0, v1) = v0 + r * (v1 - v0)`.
///
/// This is `eq(0, r) * v0 + eq(1, r) * v1 = (1 - r) * v0 + r * v1`.
pub fn fold_mle_evals<F: IsField>(
    assignment: &FieldElement<F>,
    eval0: &FieldElement<F>,
    eval1: &FieldElement<F>,
) -> FieldElement<F> {
    eval0 + &(assignment * &(eval1 - eval0))
}

/// Returns `v_0 + alpha * v_1 + ... + alpha^(n-1) * v_{n-1}` using Horner evaluation.
pub fn random_linear_combination<F: IsField>(
    values: &[FieldElement<F>],
    alpha: &FieldElement<F>,
) -> FieldElement<F> {
    horner_eval(values, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn horner_eval_works() {
        // 9 + 2x + 3x^2 at x=7 => 9 + 14 + 147 = 170 = 170 mod 101 = 69
        let coeffs = vec![FE::from(9), FE::from(2), FE::from(3)];
        let x = FE::from(7);
        let result = horner_eval(&coeffs, &x);
        let expected = FE::from(9) + FE::from(2) * &x + FE::from(3) * &x * &x;
        assert_eq!(result, expected);
    }

    #[test]
    fn eq_on_boolean_hypercube() {
        let one = FE::one();
        let zero = FE::zero();
        assert_eq!(
            eq(&[one.clone(), zero.clone()], &[one.clone(), zero.clone()]),
            FE::one()
        );
        assert_eq!(
            eq(&[one.clone(), zero.clone()], &[zero.clone(), one.clone()]),
            FE::zero()
        );
    }

    #[test]
    fn fold_mle_evals_works() {
        let v0 = FE::from(3);
        let v1 = FE::from(7);
        let r = FE::from(5);
        // 3 + 5*(7-3) = 3 + 20 = 23
        assert_eq!(fold_mle_evals(&r, &v0, &v1), FE::from(23));
    }

    #[test]
    fn random_linear_combination_works() {
        let values = vec![FE::from(1), FE::from(2), FE::from(3)];
        let alpha = FE::from(10);
        // 1 + 10*2 + 100*3 = 1 + 20 + 300 = 321 = 321 mod 101 = 18
        let result = random_linear_combination(&values, &alpha);
        let expected = FE::from(1) + FE::from(10) * FE::from(2) + FE::from(100) * FE::from(3);
        assert_eq!(result, expected);
    }
}
