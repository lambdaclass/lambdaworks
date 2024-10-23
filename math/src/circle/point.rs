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

    /// Coordinate x of the generator of the circle group.
    const CIRCLE_GENERATOR_X: FieldElement<F>;

    /// Coordinate y of the generator of the circle group.
    const CIRCLE_GENERATOR_Y: FieldElement<F>;

    const ORDER: u128;
}

impl HasCircleParams<Mersenne31Field> for Mersenne31Field {
    type FE = FieldElement<Mersenne31Field>;

    const CIRCLE_GENERATOR_X: Self::FE = Self::FE::const_from_raw(2);

    const CIRCLE_GENERATOR_Y: Self::FE = Self::FE::const_from_raw(1268011823);

    /// ORDER = 2^31
    const ORDER: u128 = 2147483648;
}

impl HasCircleParams<Degree4ExtensionField> for Degree4ExtensionField {
    type FE = FieldElement<Degree4ExtensionField>;

    const CIRCLE_GENERATOR_X: Self::FE =
        Degree4ExtensionField::const_from_coefficients(1, 0, 478637715, 513582971);

    const CIRCLE_GENERATOR_Y: Self::FE =
        Degree4ExtensionField::const_from_coefficients(992285211, 649143431, 740191619, 1186584352);

    /// ORDER = (2^31 - 1)^4 - 1
    const ORDER: u128 = 21267647892944572736998860269687930880;
}

impl<F: IsField + HasCircleParams<F>> CirclePoint<F> {
    pub fn new(x: FieldElement<F>, y: FieldElement<F>) -> Result<Self, CircleError> {
        if x.square() + y.square() == FieldElement::one() {
            Ok(Self { x, y })
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
    /// We are using -(x, y) = (x, -y), i.e. the inverse of the group opertion is conjugation
    /// because the norm of every point in the circle is one.
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

    pub const GENERATOR: Self = Self {
        x: F::CIRCLE_GENERATOR_X,
        y: F::CIRCLE_GENERATOR_Y,
    };

    /// Returns the generator of the subgroup of order n = 2^log_2_size.
    /// We are using that 2^k * g is a generator of the subgroup of order 2^{31 - k}.
    pub fn get_generator_of_subgroup(log_2_size: u32) -> Self {
        Self::GENERATOR.repeated_double(31 - log_2_size)
    }

    pub const ORDER: u128 = F::ORDER;
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
        let g = G::GENERATOR;
        let zero = G::zero();
        assert_eq!(g.clone() + zero, g)
    }

    #[test]
    fn double_equals_mul_two() {
        let g = G::GENERATOR;
        assert_eq!(g.clone().double(), G::scalar_mul(g, 2))
    }

    #[test]
    fn mul_eight_equals_double_three_times() {
        let g = G::GENERATOR;
        assert_eq!(g.clone().repeated_double(3), G::scalar_mul(g, 8))
    }

    #[test]
    fn generator_g1_has_order_two_pow_31() {
        let g = G::GENERATOR;
        let n = 31;
        assert_eq!(g.repeated_double(n), G::zero())
    }

    #[test]
    fn generator_g4_has_the_order_of_the_group() {
        let g = G4::GENERATOR;
        assert_eq!(g.scalar_mul(G4::ORDER), G4::zero())
    }

    #[test]
    fn conjugation_is_inverse_operation() {
        let g = G::GENERATOR;
        assert_eq!(g.clone() + g.conjugate(), G::zero())
    }

    #[test]
    fn subgroup_generator_has_correct_order() {
        let generator_n = G::get_generator_of_subgroup(7);
        assert_eq!(generator_n.repeated_double(7), G::zero());
    }
}
