//! FRI polynomial folding: split even/odd coefficients and combine with challenge.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

/// Fold a polynomial `p(x)` with challenge `beta`:
///
/// Given `p(x) = p_even(x^2) + x * p_odd(x^2)`, returns
/// `p_folded(x) = p_even(x) + beta * p_odd(x)`.
///
/// This halves the polynomial degree.
pub fn fold_polynomial<F: IsField>(
    poly: &Polynomial<FieldElement<F>>,
    beta: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    let coeffs = poly.coefficients();
    if coeffs.is_empty() {
        return Polynomial::zero();
    }

    let even: Vec<FieldElement<F>> = coeffs.iter().step_by(2).cloned().collect();
    let odd: Vec<FieldElement<F>> = coeffs.iter().skip(1).step_by(2).cloned().collect();

    let len = even.len().max(odd.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let e = if i < even.len() {
            even[i].clone()
        } else {
            FieldElement::zero()
        };
        let o = if i < odd.len() {
            &odd[i] * beta
        } else {
            FieldElement::zero()
        };
        result.push(e + o);
    }

    Polynomial::new(&result)
}

/// Fold a single evaluation: given `f(x)` and `f(-x)` on domain D,
/// compute `f_folded` at the corresponding point in the half-domain.
///
/// `f_folded(x^2) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x)) / (2*x)`
pub fn fold_eval<F: IsField>(
    eval: &FieldElement<F>,
    eval_sym: &FieldElement<F>,
    beta: &FieldElement<F>,
    x: &FieldElement<F>,
) -> FieldElement<F> {
    let two = FieldElement::<F>::from(2u64);
    let two_inv = two.inv().expect("2 should be invertible");
    let x_inv = x.inv().expect("domain point should be nonzero");

    // f_even = (f(x) + f(-x)) / 2
    let f_even = (eval + eval_sym) * &two_inv;
    // f_odd = (f(x) - f(-x)) / (2*x)
    let f_odd = (eval - eval_sym) * &two_inv * &x_inv;

    f_even + beta * f_odd
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 293;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn fold_polynomial_basic() {
        // p(x) = 3 + x + 2*x^2 + 7*x^3 + 3*x^4 + 5*x^5
        // even: 3, 2, 3  odd: 1, 7, 5
        // beta = 4
        // folded = (3 + 4*1), (2 + 4*7), (3 + 4*5) = 7, 30, 23
        let p = Polynomial::new(&[
            FE::new(3),
            FE::new(1),
            FE::new(2),
            FE::new(7),
            FE::new(3),
            FE::new(5),
        ]);
        let beta = FE::new(4);
        let folded = fold_polynomial(&p, &beta);
        assert_eq!(
            folded,
            Polynomial::new(&[FE::new(7), FE::new(30), FE::new(23)])
        );
    }

    #[test]
    fn fold_to_constant() {
        let p = Polynomial::new(&[FE::new(3), FE::new(7)]);
        let beta = FE::new(5);
        // folded = 3 + 5*7 = 38
        let folded = fold_polynomial(&p, &beta);
        assert_eq!(folded.degree(), 0);
        assert_eq!(folded.evaluate(&FE::new(0)), FE::new(38));
    }

    #[test]
    fn fold_eval_consistent_with_fold_polynomial() {
        // p(x) = 2 + 3x + 5x^2 + 7x^3
        let p = Polynomial::new(&[FE::new(2), FE::new(3), FE::new(5), FE::new(7)]);
        let beta = FE::new(11);
        let folded = fold_polynomial(&p, &beta);

        // Check: fold_eval at x should give folded(x^2)
        let x = FE::new(4);
        let neg_x = FE::new(0) - x.clone();
        let eval_x = p.evaluate(&x);
        let eval_neg_x = p.evaluate(&neg_x);

        let result = fold_eval(&eval_x, &eval_neg_x, &beta, &x);
        let expected = folded.evaluate(&(x.clone() * x));
        assert_eq!(result, expected);
    }
}
