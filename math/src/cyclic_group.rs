use crate::unsigned_integer::traits::IsUnsignedInteger;

pub trait IsGroup: Clone + PartialEq + Eq {
    /// Returns the neutral element of the group. The equality
    /// `neutral_element().operate_with(g) == g` must hold
    /// for every group element `g`.
    fn neutral_element() -> Self;

    /// Check if an element the neutral element.
    fn is_neutral_element(&self) -> bool {
        self == &Self::neutral_element()
    }

    /// Applies the group operation `times` times with itself
    /// The operation can be addition or multiplication depending on
    /// the notation of the particular group.
    #[cfg(not(feature = "constant-time"))]
    fn operate_with_self<T: IsUnsignedInteger>(&self, mut exponent: T) -> Self {
        let mut result = Self::neutral_element();
        let mut base = self.clone();

        while exponent != T::from(0) {
            if exponent & T::from(1) == T::from(1) {
                result = Self::operate_with(&result, &base);
            }
            exponent = exponent >> 1;
            base = Self::operate_with(&base, &base);
        }
        result
    }
    #[cfg(feature = "constant-time")]
    fn operate_with_self<T: IsUnsignedInteger>(&self, exponent: T) -> Self {
        let mut result = Self::neutral_element();
        let mut base = self.clone();

        let num_bits = std::mem::size_of::<T>() * 8;

        for i in (0..num_bits).rev() {
            let mask = T::from(1) << i;
            let bit = (exponent & mask) >> i;

            if bit == T::from(1) {
                result = Self::operate_with(&result, &base);
                base = Self::operate_with(&base, &base);
            } else {
                base = Self::operate_with(&base, &result);
                result = Self::operate_with(&result, &result);
            }
        }
        result
    }

    /// Applies the group operation between `self` and `other`.
    /// The operation can be addition or multiplication depending on
    /// the notation of the particular group.
    fn operate_with(&self, other: &Self) -> Self;

    fn neg(&self) -> Self;
}
