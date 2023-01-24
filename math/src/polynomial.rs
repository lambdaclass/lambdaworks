use super::field_element::U64FieldElement;
use std::ops;

/// Represents the polynomial c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<const ORDER: u64> {
    coefficients: Vec<U64FieldElement<ORDER>>,
}

impl<const ORDER: u64> Polynomial<ORDER> {
    /// Creates a new polynomial with the given coefficients
    pub fn new(coefficients: Vec<U64FieldElement<ORDER>>) -> Self {
        // Removes trailing zero coefficients at the end
        let mut unpadded_coefficients = coefficients
            .into_iter()
            .rev()
            .skip_while(|x| *x == U64FieldElement::from(0))
            .collect::<Vec<U64FieldElement<ORDER>>>();
        unpadded_coefficients.reverse();
        Polynomial {
            coefficients: unpadded_coefficients,
        }
    }

    pub fn new_monomial(coefficient: U64FieldElement<ORDER>, degree: usize) -> Self {
        let mut coefficients = vec![U64FieldElement::from(0); degree];
        coefficients.push(coefficient);
        Self::new(coefficients)
    }

    pub fn zero() -> Self {
        Self::new(Vec::<U64FieldElement<ORDER>>::new())
    }

    pub fn interpolate(
        xs: &[U64FieldElement<ORDER>],
        ys: &[U64FieldElement<ORDER>],
    ) -> Polynomial<ORDER> {
        let mut result = Polynomial::zero();

        for (i, y) in ys.iter().enumerate() {
            let mut y_term = Polynomial::new(vec![*y]);
            for (j, x) in xs.iter().enumerate() {
                if i != j {
                    let denominator = Polynomial::new(vec![U64FieldElement::from(1) / (xs[i] - *x)]);
                    let numerator = Polynomial::new(vec![-*x, U64FieldElement::from(1)]);
                    y_term = y_term.mul_with_ref(&(numerator * denominator));
                }
            }
            result = result + y_term;
        }
        result
    }

    pub fn evaluate(&self, x: U64FieldElement<ORDER>) -> U64FieldElement<ORDER> {
        self.coefficients
            .iter()
            .enumerate()
            .fold(U64FieldElement::from(0), |acc, (i, &c)| {
                acc + c * x.pow(i as u128)
            })
    }

    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    pub fn leading_coefficient(&self) -> U64FieldElement<ORDER> {
        if let Some(coefficient) = self.coefficients.last() {
            *coefficient
        } else {
            U64FieldElement::from(0)
        }
    }

    /// Returns coefficients of the polynomial as an array
    /// \[c_0, c_1, c_2, ..., c_n\]
    /// that represents the polynomial
    /// c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
    pub fn coefficients(&self) -> &[U64FieldElement<ORDER>] {
        &self.coefficients
    }

    /// Pads polynomial representations with minimum number of zeros to match lengths.
    fn pad_with_zero_coefficients(
        pa: &Polynomial<ORDER>,
        pb: &Polynomial<ORDER>,
    ) -> (Polynomial<ORDER>, Polynomial<ORDER>) {
        let mut pa = pa.clone();
        let mut pb = pb.clone();

        if pa.coefficients.len() > pb.coefficients.len() {
            pb.coefficients
                .resize(pa.coefficients.len(), U64FieldElement::from(0));
        } else {
            pa.coefficients
                .resize(pb.coefficients.len(), U64FieldElement::from(0));
        }
        (pa, pb)
    }

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: (quotient, remainder)
    pub fn long_division_with_remainder(self, dividend: &Self) -> (Self, Self) {
        if dividend.degree() > self.degree() {
            (Polynomial::zero(), self)
        } else {
            let mut n = self;
            let mut q: Vec<U64FieldElement<ORDER>> = vec![U64FieldElement::from(0); n.degree() + 1];
            while n != Polynomial::zero() && n.degree() >= dividend.degree() {
                let new_coefficient = n.leading_coefficient() / dividend.leading_coefficient();
                q[n.degree() - dividend.degree()] = new_coefficient;
                let d = dividend.mul_with_ref(&Polynomial::new_monomial(
                    new_coefficient,
                    n.degree() - dividend.degree(),
                ));
                n = n - d;
            }
            (Polynomial::new(q), n)
        }
    }

    pub fn div_with_ref(self, dividend: &Self) -> Self {
        let (quotient, _remainder) = self.long_division_with_remainder(dividend);
        quotient
    }

    pub fn mul_with_ref(&self, factor: &Self) -> Self {
        let degree = self.degree() + factor.degree();
        let mut coefficients = vec![U64FieldElement::from(0); degree + 1];

        if self.coefficients.is_empty() || factor.coefficients.is_empty() {
            Polynomial::new(vec![U64FieldElement::from(0)])
        } else {
            for i in 0..=factor.degree() {
                for j in 0..=self.degree() {
                    coefficients[i + j] += factor.coefficients[i] * self.coefficients[j];
                }
            }
            Polynomial::new(coefficients)
        }
    }
}

impl<const ORDER: u64> ops::Add<&Polynomial<ORDER>> for &Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn add(self, a_polynomial: &Polynomial<ORDER>) -> Self::Output {
        let (pa, pb) = Polynomial::pad_with_zero_coefficients(self, a_polynomial);
        let iter_coeff_pa = pa.coefficients.iter();
        let iter_coeff_pb = pb.coefficients.iter();
        let new_coefficients = iter_coeff_pa.zip(iter_coeff_pb).map(|(&x, &y)| x + y);

        Polynomial::new(new_coefficients.collect())
    }
}

impl<const ORDER: u64> ops::Add<Polynomial<ORDER>> for Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn add(self, a_polynomial: Polynomial<ORDER>) -> Polynomial<ORDER> {
        &self + &a_polynomial
    }
}

impl<const ORDER: u64> ops::Add<&Polynomial<ORDER>> for Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn add(self, a_polynomial: &Polynomial<ORDER>) -> Polynomial<ORDER> {
        &self + a_polynomial
    }
}

impl<const ORDER: u64> ops::Add<Polynomial<ORDER>> for &Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn add(self, a_polynomial: Polynomial<ORDER>) -> Polynomial<ORDER> {
        self + &a_polynomial
    }
}
impl<const ORDER: u64> ops::Neg for Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn neg(self) -> Polynomial<ORDER> {
        Polynomial::new(self.coefficients.iter().map(|&x| -x).collect())
    }
}

impl<const ORDER: u64> ops::Sub<Polynomial<ORDER>> for Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn sub(self, substrahend: Polynomial<ORDER>) -> Polynomial<ORDER> {
        self + (-substrahend)
    }
}

impl<const ORDER: u64> ops::Div<Polynomial<ORDER>> for Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;

    fn div(self, dividend: Polynomial<ORDER>) -> Polynomial<ORDER> {
        self.div_with_ref(&dividend)
    }
}

impl<const ORDER: u64> ops::Mul<Polynomial<ORDER>> for Polynomial<ORDER> {
    type Output = Polynomial<ORDER>;
    fn mul(self, dividend: Polynomial<ORDER>) -> Polynomial<ORDER> {
        self.mul_with_ref(&dividend)
    }
}

#[cfg(test)]
mod tests {
    // Some of these tests work when the finite field has order greater than 2.
    use super::*;
    const ORDER: u64 = 23;
    type FE = U64FieldElement<ORDER>;

    fn polynomial_a() -> Polynomial<ORDER> {
        Polynomial::new(vec![FE::from(1), FE::from(2), FE::from(3)])
    }

    fn polynomial_minus_a() -> Polynomial<ORDER> {
        Polynomial::new(vec![
            FE::from(ORDER - 1),
            FE::from(ORDER - 2),
            FE::from(ORDER - 3),
        ])
    }

    fn polynomial_b() -> Polynomial<ORDER> {
        Polynomial::new(vec![FE::from(3), FE::from(4), FE::from(5)])
    }

    fn polynomial_a_plus_b() -> Polynomial<ORDER> {
        Polynomial::new(vec![FE::from(4), FE::from(6), FE::from(8)])
    }

    fn polynomial_b_minus_a() -> Polynomial<ORDER> {
        Polynomial::new(vec![FE::from(2), FE::from(2), FE::from(2)])
    }

    #[test]
    fn adding_a_and_b_equals_a_plus_b() {
        assert_eq!(polynomial_a() + polynomial_b(), polynomial_a_plus_b());
    }

    #[test]
    fn adding_a_and_a_plus_b_does_not_equal_b() {
        assert_ne!(polynomial_a() + polynomial_a_plus_b(), polynomial_b());
    }

    #[test]
    fn add_5_to_0_is_5() {
        let p1 = Polynomial::new(vec![FE::from(5)]);
        let p2 = Polynomial::new(vec![FE::from(0)]);
        assert_eq!(p1 + p2, Polynomial::new(vec![FE::from(5)]));
    }

    #[test]
    fn add_0_to_5_is_5() {
        let p1 = Polynomial::new(vec![FE::from(5)]);
        let p2 = Polynomial::new(vec![FE::from(0)]);
        assert_eq!(p2 + p1, Polynomial::new(vec![FE::from(5)]));
    }

    #[test]
    fn negating_0_returns_0() {
        let p1 = Polynomial::new(vec![FE::from(0)]);
        assert_eq!(-p1, Polynomial::new(vec![FE::from(0)]));
    }

    #[test]
    fn negating_a_is_equal_to_minus_a() {
        assert_eq!(-polynomial_a(), polynomial_minus_a());
    }

    #[test]
    fn negating_a_is_not_equal_to_a() {
        assert_ne!(-polynomial_a(), polynomial_a());
    }

    #[test]
    fn substracting_5_5_gives_0() {
        let p1 = Polynomial::new(vec![FE::from(5)]);
        let p2 = Polynomial::new(vec![FE::from(5)]);
        let p3 = Polynomial::new(vec![FE::from(0)]);
        assert_eq!(p1 - p2, p3);
    }

    #[test]
    fn substracting_b_and_a_equals_b_minus_a() {
        assert_eq!(polynomial_b() - polynomial_a(), polynomial_b_minus_a());
    }

    #[test]
    fn constructor_removes_zeros_at_the_end_of_polynomial() {
        let p1 = Polynomial::new(vec![FE::from(3), FE::from(4), FE::from(0)]);
        assert_eq!(p1.coefficients, vec![FE::from(3), FE::from(4)]);
    }

    #[test]
    fn pad_with_zero_coefficients_returns_polynomials_with_zeros_until_matching_size() {
        let p1 = Polynomial::new(vec![FE::from(3), FE::from(4)]);
        let p2 = Polynomial::new(vec![FE::from(3)]);

        assert_eq!(p2.coefficients, vec![FE::from(3)]);
        let (pp1, pp2) = Polynomial::pad_with_zero_coefficients(&p1, &p2);
        assert_eq!(pp1, p1);
        assert_eq!(pp2.coefficients, vec![FE::from(3), FE::from(0)]);
    }

    #[test]
    fn multiply_5_and_0_is_0() {
        let p1 = Polynomial::new(vec![FE::from(5)]);
        let p2 = Polynomial::new(vec![FE::from(0)]);
        assert_eq!(p1 * p2, Polynomial::new(vec![FE::from(0)]));
    }

    #[test]
    fn multiply_0_and_x_is_0() {
        let p1 = Polynomial::new(vec![FE::from(0)]);
        let p2 = Polynomial::new(vec![FE::from(0), FE::from(1)]);
        assert_eq!(p1 * p2, Polynomial::new(vec![FE::from(0)]));
    }

    #[test]
    fn multiply_2_by_3_is_6() {
        let p1 = Polynomial::new(vec![FE::from(2)]);
        let p2 = Polynomial::new(vec![FE::from(3)]);
        assert_eq!(p1 * p2, Polynomial::new(vec![FE::from(6)]));
    }

    #[test]
    fn multiply_2xx_3x_3_times_x_4() {
        let p1 = Polynomial::new(vec![FE::from(3), FE::from(3), FE::from(2)]);
        let p2 = Polynomial::new(vec![FE::from(4), FE::from(1)]);
        assert_eq!(
            p1 * p2,
            Polynomial::new(vec![FE::from(12), FE::from(15), FE::from(11), FE::from(2)])
        );
    }

    #[test]
    fn multiply_x_4_times_2xx_3x_3() {
        let p1 = Polynomial::new(vec![FE::from(3), FE::from(3), FE::from(2)]);
        let p2 = Polynomial::new(vec![FE::from(4), FE::from(1)]);
        assert_eq!(
            p2 * p1,
            Polynomial::new(vec![FE::from(12), FE::from(15), FE::from(11), FE::from(2)])
        );
    }

    #[test]
    fn division_works() {
        let p1 = Polynomial::new(vec![FE::from(1), FE::from(3)]);
        let p2 = Polynomial::new(vec![FE::from(1), FE::from(3)]);
        let p3 = p1.mul_with_ref(&p2);
        assert_eq!(p3 / p2, p1);
    }

    #[test]
    fn division_by_zero_degree_polynomial_works() {
        let four = FE::from(4);
        let two = FE::from(2);
        let p1 = Polynomial::new(vec![four, four]);
        let p2 = Polynomial::new(vec![two]);
        assert_eq!(Polynomial::new(vec![two, two]), p1 / p2);
    }

    #[test]
    fn evaluate_constant_polynomial_returns_constant() {
        let three = FE::from(3);
        let p = Polynomial::new(vec![three]);
        assert_eq!(p.evaluate(FE::from(10)), three);
    }

    #[test]
    fn create_degree_0_new_monomial() {
        assert_eq!(
            Polynomial::new_monomial(FE::from(3), 0),
            Polynomial::new(vec![FE::from(3)])
        );
    }

    #[test]
    fn zero_poly_evals_0_in_3() {
        assert_eq!(
            Polynomial::new_monomial(FE::from(0), 0).evaluate(FE::from(3)),
            FE::from(0)
        );
    }

    #[test]
    fn evaluate_degree_1_new_monomial() {
        let two = FE::from(2);
        let four = FE::from(4);
        let p = Polynomial::new_monomial(two, 1);
        assert_eq!(p.evaluate(two), four);
    }

    #[test]
    fn evaluate_degree_2_monomyal() {
        let two = FE::from(2);
        let eight = FE::from(8);
        let p = Polynomial::new_monomial(two, 2);
        assert_eq!(p.evaluate(two), eight);
    }

    #[test]
    fn evaluate_3_term_polynomial() {
        let p = Polynomial::new(vec![FE::from(3), -FE::from(2), FE::from(4)]);
        assert_eq!(p.evaluate(FE::from(2)), FE::from(15));
    }

    #[test]
    fn simple_interpolating_polynomial_by_hand_works() {
        let denominator = Polynomial::new(vec![FE::from(1) / (FE::from(2) - FE::from(4))]);
        let numerator = Polynomial::new(vec![-FE::from(4), FE::from(1)]);
        let interpolating = numerator * denominator;
        assert_eq!(
            (FE::from(2) - FE::from(4)) * (FE::from(1) / (FE::from(2) - FE::from(4))),
            FE::from(1)
        );
        assert_eq!(interpolating.evaluate(FE::from(2)), FE::from(1));
        assert_eq!(interpolating.evaluate(FE::from(4)), FE::from(0));
    }

    #[test]
    fn interpolate_x_2_y_3() {
        let p = Polynomial::interpolate(&[FE::from(2)], &[FE::from(3)]);
        assert_eq!(FE::from(3), p.evaluate(FE::from(2)));
    }

    #[test]
    fn interpolate_x_0_2_y_3_4() {
        let p = Polynomial::interpolate(&[FE::from(0), FE::from(2)], &[FE::from(3), FE::from(4)]);
        assert_eq!(FE::from(3), p.evaluate(FE::from(0)));
        assert_eq!(FE::from(4), p.evaluate(FE::from(2)));
    }

    #[test]
    fn interpolate_x_2_5_7_y_10_19_43() {
        let p = Polynomial::interpolate(
            &[FE::from(2), FE::from(5), FE::from(7)],
            &[FE::from(10), FE::from(19), FE::from(43)],
        );

        assert_eq!(FE::from(10), p.evaluate(FE::from(2)));
        assert_eq!(FE::from(19), p.evaluate(FE::from(5)));
        assert_eq!(FE::from(43), p.evaluate(FE::from(7)));
    }

    #[test]
    fn interpolate_x_0_0_y_1_1() {
        let p = Polynomial::interpolate(&[FE::from(0), FE::from(1)], &[FE::from(0), FE::from(1)]);

        assert_eq!(FE::from(0), p.evaluate(FE::from(0)));
        assert_eq!(FE::from(1), p.evaluate(FE::from(1)));
    }

    #[test]
    fn interpolate_x_0_y_0() {
        let p = Polynomial::interpolate(&[FE::from(0)], &[FE::from(0)]);
        assert_eq!(FE::from(0), p.evaluate(FE::from(0)));
    }
}
