use core::iter::Sum;
use core::ops::Add;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// A projective fraction `numerator / denominator`.
///
/// Kept in projective form to avoid field inversions during accumulation.
#[derive(Debug, Clone, PartialEq)]
pub struct Fraction<F: IsField> {
    pub numerator: FieldElement<F>,
    pub denominator: FieldElement<F>,
}

impl<F: IsField> Fraction<F> {
    pub fn new(numerator: FieldElement<F>, denominator: FieldElement<F>) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

/// `a/b + c/d = (a*d + b*c) / (b*d)`
impl<F: IsField> Add for Fraction<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Fraction {
            numerator: &self.numerator * &rhs.denominator + &self.denominator * &rhs.numerator,
            denominator: self.denominator * rhs.denominator,
        }
    }
}

impl<F: IsField> Sum for Fraction<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or_else(|| Fraction {
            numerator: FieldElement::zero(),
            denominator: FieldElement::one(),
        })
    }
}

/// Represents the fraction `1 / x`. Adding two reciprocals gives a `Fraction`.
pub struct Reciprocal<F: IsField> {
    pub x: FieldElement<F>,
}

impl<F: IsField> Reciprocal<F> {
    pub fn new(x: FieldElement<F>) -> Self {
        Self { x }
    }
}

/// `1/a + 1/b = (a + b) / (a * b)`
impl<F: IsField> Add for Reciprocal<F> {
    type Output = Fraction<F>;

    fn add(self, rhs: Self) -> Fraction<F> {
        Fraction {
            numerator: &self.x + &rhs.x,
            denominator: self.x * rhs.x,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn fraction_addition() {
        // 1/3 + 2/6 = (1*6 + 3*2) / (3*6) = 12/18 = 2/3
        let a = Fraction::new(FE::from(1), FE::from(3));
        let b = Fraction::new(FE::from(2), FE::from(6));
        let result = a + b;
        // Check numerator/denominator = 2/3
        let expected = FE::from(2) * FE::from(3).inv().unwrap();
        let actual = result.numerator * result.denominator.inv().unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn fraction_sum() {
        // 1/2 + 1/3 + 1/6 = 1
        let fractions = vec![
            Fraction::new(FE::from(1), FE::from(2)),
            Fraction::new(FE::from(1), FE::from(3)),
            Fraction::new(FE::from(1), FE::from(6)),
        ];
        let result: Fraction<F> = fractions.into_iter().sum();
        let actual = result.numerator * result.denominator.inv().unwrap();
        assert_eq!(actual, FE::one());
    }

    #[test]
    fn reciprocal_addition() {
        // 1/3 + 1/5 = (3+5)/(3*5) = 8/15
        let a = Reciprocal::new(FE::from(3));
        let b = Reciprocal::new(FE::from(5));
        let result = a + b;
        assert_eq!(result.numerator, FE::from(8));
        assert_eq!(result.denominator, FE::from(15));
    }
}
