use crate::unsigned_integer::traits::IsUnsignedInteger;

pub trait IsGroup: Clone + PartialEq + Eq {
    /// Returns the neutral element of the group. The equality
    /// `neutral_element().operate_with(g) == g` must hold
    /// for every group element `g`.
    fn neutral_element() -> Self;

    /// Check if an element the neutral element.
    #[inline]
    fn is_neutral_element(&self) -> bool {
        self == &Self::neutral_element()
    }

    /// Applies the group operation `times` times with itself
    /// The operation can be addition or multiplication depending on
    /// the notation of the particular group.
    fn operate_with_self<T: IsUnsignedInteger>(&self, mut exponent: T) -> Self {
        let zero = T::from(0);
        let one = T::from(1);

        if exponent == zero {
            return Self::neutral_element();
        }

        let mut result = Self::neutral_element();
        let mut base = self.clone();

        loop {
            if exponent & one == one {
                result = result.operate_with(&base);
            }
            exponent >>= 1;
            if exponent == zero {
                break;
            }
            base = base.double();
        }
        result
    }

    /// Applies the group operation between `self` and `other`.
    /// The operation can be addition or multiplication depending on
    /// the notation of the particular group.
    fn operate_with(&self, other: &Self) -> Self;

    /// Provides the inverse of the group element.
    /// This is the unique y such that for any x
    /// x.operate_with(y) returns the neutral element
    fn neg(&self) -> Self;

    /// Returns the double of the element (self + self)
    #[inline]
    fn double(&self) -> Self {
        self.operate_with(self)
    }
}
