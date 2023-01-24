use crate::algebraic_element::{Field, FieldElement};
use std::marker::PhantomData;

pub trait QuadraticNonResidue<FieldElement> {
    fn quadratic_non_residue() -> FieldElement;
}

#[derive(Debug, Clone)]
pub struct QuadraticFieldExtensionBackend<T> {
    phantom: PhantomData<T>
}

impl<T, Set, Backend> Field<[FieldElement<Set, Backend>; 2]> for QuadraticFieldExtensionBackend<T>
where
    FieldElement<Set, Backend>: Clone,
    Backend: Field<Set>,
    T: QuadraticNonResidue<FieldElement<Set, Backend>>
{
    fn add(a: &[FieldElement<Set, Backend>; 2], b: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2] {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    fn mul(a: &[FieldElement<Set, Backend>; 2], b: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2]{
        let q = T::quadratic_non_residue();
        // (a0 + a1 t) (b0 + b1 t) = a0 b0 + a1 b1 q + t( a0 b1 + a1 b0 )
        [&a[0] * &b[0] + &a[1] * &b[1] * q, &a[0] * &b[1] + &a[1] * &b[0]]
    }

    fn pow(a: &[FieldElement<Set, Backend>; 2], mut exponent: u128) -> [FieldElement<Set, Backend>; 2]{
        let mut result = Self::one();
        let mut base = (*a).clone();

        while exponent > 0 {
            if exponent & 1 == 1 {
                result = Self::mul(&result, &base);
            }
            exponent >>= 1;
            base = Self::mul(&base, &base);
        }
        result
    }

    fn sub(a: &[FieldElement<Set, Backend>; 2], b: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2]{
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    fn neg(a: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2]{
        [-&a[0], -&a[1]]
    }

    fn inv(a: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2] {
        let inv_norm = (a[0].pow(2) - T::quadratic_non_residue() * a[1].pow(2)).inv();
        [&a[0] * &inv_norm, - &a[1] * inv_norm]
    }

    fn div(a: &[FieldElement<Set, Backend>; 2], b: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2]{
        Self::mul(&a, &Self::inv(b))
    }

    fn eq(a: &[FieldElement<Set, Backend>; 2], b: &[FieldElement<Set, Backend>; 2]) -> bool{
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> [FieldElement<Set, Backend>; 2]{
        [FieldElement::zero(), FieldElement::zero()]
    }

    fn one() -> [FieldElement<Set, Backend>; 2]{
        [FieldElement::one(), FieldElement::zero()]
    }

    fn representative(a: &[FieldElement<Set, Backend>; 2]) -> [FieldElement<Set, Backend>; 2] {
        (*a).clone()
    }
}

impl<T, Set, Backend> FieldElement<[FieldElement<Set, Backend>; 2], QuadraticFieldExtensionBackend<T>>
where
    FieldElement<Set, Backend>: Clone,
    Backend: Field<Set>,
    T: QuadraticNonResidue<FieldElement<Set, Backend>>
{
    pub fn new_base(a: &FieldElement<Set, Backend>) -> Self {
        FieldElement::from(&[a.clone(), FieldElement::<Set,Backend>::zero()])
    }
}


#[cfg(test)]
mod tests {
    
    use crate::{config::ORDER_P, field_element::U64FieldElement};

    use super::*;

    #[derive(Debug)]
    struct MyQuadraticNonResidue;
    impl QuadraticNonResidue<U64FieldElement<ORDER_P>> for MyQuadraticNonResidue {
        fn quadratic_non_residue() -> U64FieldElement<ORDER_P> {
            -FieldElement::one()
        }
    }
    type MyFieldExtensionBackend = QuadraticFieldExtensionBackend<MyQuadraticNonResidue>;
    type FE = U64FieldElement<ORDER_P>;

    #[allow(clippy::upper_case_acronyms)]
    type FEE =FieldElement<[FE; 2], MyFieldExtensionBackend>;

    #[test]
    fn test_add_1() {
        let a = FEE::from([FE::from(0), FE::from(3)]);
        let b = FEE::from([-FE::from(2), FE::from(8)]);
        let expected_result = FEE::from([FE::from(57), FE::from(11)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_add_2() {
        let a = FEE::from([FE::from(12), FE::from(5)]);
        let b = FEE::from([-FE::from(4), FE::from(2)]);
        let expected_result = FEE::from([FE::from(8), FE::from(7)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_1() {
        let a = FEE::from([FE::from(0), FE::from(3)]);
        let b = FEE::from([-FE::from(2), FE::from(8)]);
        let expected_result = FEE::from([FE::from(2), FE::from(54)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_sub_2() {
        let a = FEE::from([FE::from(12), FE::from(5)]);
        let b = FEE::from([-FE::from(4), FE::from(2)]);
        let expected_result = FEE::from([FE::from(16), FE::from(3)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_1() {
        let a = FEE::from([FE::from(0), FE::from(3)]);
        let b = FEE::from([-FE::from(2), FE::from(8)]);
        let expected_result = FEE::from([FE::from(35), FE::from(53)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_mul_2() {
        let a = FEE::from([FE::from(12), FE::from(5)]);
        let b = FEE::from([-FE::from(4), FE::from(2)]);
        let expected_result = FEE::from([FE::from(1), FE::from(4)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_div_1() {
        let a = FEE::from([FE::from(0), FE::from(3)]);
        let b = FEE::from([-FE::from(2), FE::from(8)]);
        let expected_result = FEE::from([FE::from(42), FE::from(19)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_div_2() {
        let a = FEE::from([FE::from(12), FE::from(5)]);
        let b = FEE::from([-FE::from(4), FE::from(2)]);
        let expected_result = FEE::from([FE::from(4), FE::from(45)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_pow_1() {
        let a = FEE::from([FE::from(0), FE::from(3)]);
        let b = 5;
        let expected_result = FEE::from([FE::from(0), FE::from(7)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_pow_2() {
        let a = FEE::from([FE::from(12), FE::from(5)]);
        let b = 8;
        let expected_result = FEE::from([FE::from(52), FE::from(35)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_inv_1() {
        let a = FEE::from([FE::from(0), FE::from(3)]);
        let expected_result = FEE::from([FE::from(0), FE::from(39)]);
        assert_eq!(a.inv(), expected_result);
    }

    #[test]
    fn test_inv() {
        let a = FEE::from([FE::from(12), FE::from(5)]);
        let expected_result = FEE::from([FE::from(28), FE::from(8)]);
        assert_eq!(a.inv(), expected_result);
    }
}
