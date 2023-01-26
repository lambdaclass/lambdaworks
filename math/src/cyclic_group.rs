use std::ops::Mul;

pub trait HasCyclicBilinearGroupStructure: Clone {
    type PairingOutput: Mul<Output = Self::PairingOutput> + PartialEq + Eq;
    /// Returns a generator of the group. Every element of the group
    /// has to be of the form `operate_with_self(generator(), k)` for some `k`.
    fn generator() -> Self;
    /// Returns the neutral element of the group. The equality
    /// `neutral_element().operate_with(g) == g` must hold
    /// for every group element `g`.
    fn neutral_element() -> Self;
    /// Applies the group operation `times` times with itself
    /// The operation can be addition or multiplication depending on
    /// the notation of the particular group.
    fn operate_with_self(&self, mut exponent: u128) -> Self {
        let mut result = Self::neutral_element();
        let mut base = self.clone();

        while exponent > 0 {
            if exponent & 1 == 1 {
                result = Self::operate_with(&result, &base);
            }
            exponent >>= 1;
            base = Self::operate_with(&base, &base);
        }
        result
    }
    /// Applies the group operation between `self` and `other`.
    /// Thperation can be addition or multiplication depending on
    /// the notation of the particular group.
    fn operate_with(&self, other: &Self) -> Self;
    /// A bilinear map.
    fn pairing(&self, other: &Self) -> Self::PairingOutput;
}
