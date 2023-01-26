use std::fmt::Debug;

pub trait HasFieldOperations: Debug {
    type BaseType: Clone + Debug;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn pow(a: &Self::BaseType, mut exponent: u128) -> Self::BaseType {
        let mut result = Self::one();
        let mut base = a.clone();

        while exponent > 0 {
            if exponent & 1 == 1 {
                result = Self::mul(&result, &base);
            }
            exponent >>= 1;
            base = Self::mul(&base, &base);
        }
        result
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn neg(a: &Self::BaseType) -> Self::BaseType;

    fn inv(a: &Self::BaseType) -> Self::BaseType;

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType;

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool;

    fn zero() -> Self::BaseType;

    fn one() -> Self::BaseType;

    fn representative(a: &Self::BaseType) -> Self::BaseType;

    fn from_u64(x: u64) -> Self::BaseType;
}
