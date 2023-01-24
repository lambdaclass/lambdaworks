use std::marker::PhantomData;

use crate::{algebraic_element::{FieldOperations, FieldElement}};

pub trait HasQuadraticNonResidue<F: FieldOperations> {
    fn residue() -> FieldElement<F>;
}

#[derive(Debug, Clone)]
pub struct QuadraticExtensionField<F, Q> 
where
    F: FieldOperations,
    Q: HasQuadraticNonResidue<F>
{
    field: PhantomData<F>,
    non_residue: PhantomData<Q>
}

impl<F, Q> FieldOperations for QuadraticExtensionField<F, Q>
where
    F: FieldOperations + Clone, 
    Q: HasQuadraticNonResidue<F> + Clone
{
    type BaseType = [FieldElement<F>; 2];

    fn add(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    fn mul(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2]{
        let q = Q::residue();
        // (a0 + a1 t) (b0 + b1 t) = a0 b0 + a1 b1 q + t( a0 b1 + a1 b0 )
        [&a[0] * &b[0] + &a[1] * &b[1] * q, &a[0] * &b[1] + &a[1] * &b[0]]
    }
    
    fn sub(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2]{
        [&a[0] - &b[0], &a[1] - &b[1]]
    }

    fn neg(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2]{
        [-&a[0], -&a[1]]
    }

    fn inv(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        let inv_norm = (a[0].pow(2) - Q::residue() * a[1].pow(2)).inv();
        [&a[0] * &inv_norm, - &a[1] * inv_norm]
    }

    fn div(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2]{
        Self::mul(&a, &Self::inv(b))
    }

    fn eq(a: &[FieldElement<F>; 2], b: &[FieldElement<F>; 2]) -> bool{
        a[0] == b[0] && a[1] == b[1]
    }

    fn zero() -> [FieldElement<F>; 2]{
        [FieldElement::zero(), FieldElement::zero()]
    }

    fn one() -> [FieldElement<F>; 2]{
        [FieldElement::one(), FieldElement::zero()]
    }

    fn representative(a: &[FieldElement<F>; 2]) -> [FieldElement<F>; 2] {
        a.clone()
    }
}

impl<F: FieldOperations + Clone, Q: HasQuadraticNonResidue<F> + Clone> FieldElement<QuadraticExtensionField<F, Q>>
{
    pub fn new_base(a: &FieldElement<F>) -> Self {
        FieldElement::new([a.clone(), FieldElement::<F>::zero()])
    }
}

pub type QuadraticExtensionFieldElement<F, Q> = FieldElement<QuadraticExtensionField<F, Q>>;


#[cfg(test)]
mod tests {
    
    use crate::{field_element::{U64FieldElement, U64PrimeField}, config::ORDER_P};

    use super::*;

    #[derive(Debug, Clone)]
    struct MyQuadraticNonResidue;
    impl HasQuadraticNonResidue<U64PrimeField<ORDER_P>> for MyQuadraticNonResidue {
        fn residue() -> FieldElement<U64PrimeField<ORDER_P>> {
            -FieldElement::one()
        }
    }

    type FE = U64FieldElement<ORDER_P>;
    type MyFieldExtensionBackend = QuadraticExtensionField<U64PrimeField<ORDER_P>, MyQuadraticNonResidue>;
    #[allow(clippy::upper_case_acronyms)]
    type FEE =FieldElement<MyFieldExtensionBackend>;

    #[test]
    fn test_add_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(57), FE::new(11)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_add_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(8), FE::new(7)]);
        assert_eq!(a + b, expected_result);
    }

    #[test]
    fn test_sub_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(2), FE::new(54)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_sub_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(16), FE::new(3)]);
        assert_eq!(a - b, expected_result);
    }

    #[test]
    fn test_mul_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(35), FE::new(53)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_mul_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(1), FE::new(4)]);
        assert_eq!(a * b, expected_result);
    }

    #[test]
    fn test_div_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = FEE::new([-FE::new(2), FE::new(8)]);
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_div_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = FEE::new([-FE::new(4), FE::new(2)]);
        let expected_result = FEE::new([FE::new(4), FE::new(45)]);
        assert_eq!(a / b, expected_result);
    }

    #[test]
    fn test_pow_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let b = 5;
        let expected_result = FEE::new([FE::new(0), FE::new(7)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_pow_2() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let b = 8;
        let expected_result = FEE::new([FE::new(52), FE::new(35)]);
        assert_eq!(a.pow(b), expected_result);
    }

    #[test]
    fn test_inv_1() {
        let a = FEE::new([FE::new(0), FE::new(3)]);
        let expected_result = FEE::new([FE::new(0), FE::new(39)]);
        assert_eq!(a.inv(), expected_result);
    }

    #[test]
    fn test_inv() {
        let a = FEE::new([FE::new(12), FE::new(5)]);
        let expected_result = FEE::new([FE::new(28), FE::new(8)]);
        assert_eq!(a.inv(), expected_result);
    }
}
