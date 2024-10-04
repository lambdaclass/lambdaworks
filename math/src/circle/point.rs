use super::errors::CircleError;
use crate::field::traits::IsField;
use crate::field::{
    element::FieldElement,
    fields::mersenne31::{extensions::Degree4ExtensionField, field::Mersenne31Field},
};
use core::ops::Add;

#[derive(Debug, Clone)]
pub struct CirclePoint<F: IsField> {
    pub x: FieldElement<F>,
    pub y: FieldElement<F>,
}

pub trait HasCircleParams<F: IsField> {
    type FE;

    fn circle_generator() -> (FieldElement<F>, FieldElement<F>);
    const ORDER: u128;
}

impl HasCircleParams<Mersenne31Field> for Mersenne31Field {
    type FE = FieldElement<Mersenne31Field>;

    // This could be a constant instead of a function
    fn circle_generator() -> (Self::FE, Self::FE) {
        (Self::FE::from(&2), Self::FE::from(&1268011823))
    }

    /// ORDER = 2^31
    const ORDER: u128 = 2147483648;
}

impl HasCircleParams<Degree4ExtensionField> for Degree4ExtensionField {
    type FE = FieldElement<Degree4ExtensionField>;

    // This could be a constant instead of a function
    fn circle_generator() -> (
        FieldElement<Degree4ExtensionField>,
        FieldElement<Degree4ExtensionField>,
    ) {
        (
            Degree4ExtensionField::from_coeffcients(
                FieldElement::<Mersenne31Field>::one(),
                FieldElement::<Mersenne31Field>::zero(),
                FieldElement::<Mersenne31Field>::from(&478637715),
                FieldElement::<Mersenne31Field>::from(&513582971),
            ),
            Degree4ExtensionField::from_coeffcients(
                FieldElement::<Mersenne31Field>::from(992285211),
                FieldElement::<Mersenne31Field>::from(649143431),
                FieldElement::<Mersenne31Field>::from(&740191619),
                FieldElement::<Mersenne31Field>::from(&1186584352),
            ),
        )
    }

    /// ORDER = (2^31 - 1)^4 - 1
    const ORDER: u128 = 21267647892944572736998860269687930880;
}

impl<F: IsField + HasCircleParams<F>> CirclePoint<F> {
    pub fn new(x: FieldElement<F>, y: FieldElement<F>) -> Result<Self, CircleError> {
        if x.square() + y.square() == FieldElement::one() {
            Ok(CirclePoint { x, y })
        } else {
            Err(CircleError::InvalidValue)
        }
    }

    /// Neutral element of the Circle group (with additive notation).
    pub fn zero() -> Self {
        Self::new(FieldElement::one(), FieldElement::zero()).unwrap()
    }

    /// Computes (a0, a1) + (b0, b1) = (a0 * b0 - a1 * b1, a0 * b1  + a1 * b0)
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Self, b: Self) -> Self {
        let x = &a.x * &b.x - &a.y * &b.y;
        let y = a.x * b.y + a.y * b.x;
        CirclePoint { x, y }
    }

    /// Computes n * (x, y) = (x ,y) + ... + (x, y) n-times.
    pub fn scalar_mul(self, mut scalar: u128) -> Self {
        let mut res = Self::zero();
        let mut cur = self;
        loop {
            if scalar == 0 {
                return res;
            }
            if scalar & 1 == 1 {
                res = res + cur.clone();
            }
            cur = cur.double();
            scalar >>= 1;
        }
    }

    /// Computes 2(x, y) = (2x^2 - 1, 2xy).
    pub fn double(self) -> Self {
        Self::new(
            self.x.square().double() - FieldElement::one(),
            self.x.double() * self.y,
        )
        .unwrap()
    }

    /// Computes 2^n * (x, y).
    pub fn repeated_double(self, n: u32) -> Self {
        let mut res = self;
        for _ in 0..n {
            res = res.double();
        }
        res
    }

    /// Computes the inverse of the point.
    /// We are using -(x, y) = (x, -y), i.e. the inverse of the group opertion is conjugation.
    pub fn conjugate(self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    pub fn antipode(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }

    pub fn eq(a: Self, b: Self) -> bool {
        a.x == b.x && a.y == b.y
    }

    pub fn generator() -> Self {
        CirclePoint::new(F::circle_generator().0, F::circle_generator().1).unwrap()
    }

    /// Returns the generator of the subgroup of order n = 2^log_2_size.
    /// We are using that 2^k * g is a generator of the subgroup of order 2^{31 - k}.
    pub fn get_generator_of_subgroup(log_2_size: u32) -> Self {
        Self::generator().repeated_double(31 - log_2_size)
    }

    pub fn group_order() -> u128 {
        F::ORDER
    }
}

impl<F: IsField + HasCircleParams<F>> PartialEq for CirclePoint<F> {
    fn eq(&self, other: &Self) -> bool {
        CirclePoint::eq(self.clone(), other.clone())
    }
}

impl<F: IsField + HasCircleParams<F>> Add for CirclePoint<F> {
    type Output = CirclePoint<F>;
    fn add(self, other: Self) -> Self {
        CirclePoint::add(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type F = Mersenne31Field;
    type FE = FieldElement<F>;
    type G = CirclePoint<F>;

    type Fp4 = Degree4ExtensionField;
    type Fp4E = FieldElement<Fp4>;
    type G4 = CirclePoint<Fp4>;

    #[test]
    fn create_new_valid_g_point() {
        let valid_point = G::new(FE::one(), FE::zero()).unwrap();
        let expected = G {
            x: FE::one(),
            y: FE::zero(),
        };
        assert_eq!(valid_point, expected)
    }

    #[test]
    fn create_new_valid_g4_point() {
        let valid_point = G4::new(Fp4E::one(), Fp4E::zero()).unwrap();
        let expected = G4 {
            x: Fp4E::one(),
            y: Fp4E::zero(),
        };
        assert_eq!(valid_point, expected)
    }

    #[test]
    fn create_new_invalid_circle_point() {
        let invalid_point = G::new(FE::one(), FE::one());
        assert!(invalid_point.is_err())
    }

    #[test]
    fn create_new_invalid_g4_circle_point() {
        let invalid_point = G4::new(Fp4E::one(), Fp4E::one());
        assert!(invalid_point.is_err())
    }

    #[test]
    fn zero_plus_zero_is_zero() {
        let a = G::zero();
        let b = G::zero();
        assert_eq!(a + b, G::zero())
    }

    #[test]
    fn generator_plus_zero_is_generator() {
        let g = G::generator();
        let zero = G::zero();
        assert_eq!(g.clone() + zero, g)
    }

    #[test]
    fn double_equals_mul_two() {
        let g = G::generator();
        assert_eq!(g.clone().double(), G::scalar_mul(g, 2))
    }

    #[test]
    fn mul_eight_equals_double_three_times() {
        let g = G::generator();
        assert_eq!(g.clone().repeated_double(3), G::scalar_mul(g, 8))
    }

    #[test]
    fn generator_g1_has_order_two_pow_31() {
        let g = G::generator();
        let n = 31;
        assert_eq!(g.repeated_double(n), G::zero())
    }

    #[test]
    fn generator_g4_has_the_order_of_the_group() {
        let g = G4::generator();
        assert_eq!(g.scalar_mul(G4::group_order()), G4::zero())
    }

    #[test]
    fn conjugation_is_inverse_operation() {
        let g = G::generator();
        assert_eq!(g.clone() + g.conjugate(), G::zero())
    }

    #[test]
    fn subgroup_generator_has_correct_order() {
        let generator_n = G::get_generator_of_subgroup(7);
        assert_eq!(generator_n.repeated_double(7), G::zero());
    }
}
