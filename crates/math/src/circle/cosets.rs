extern crate alloc;
use crate::circle::point::CirclePoint;
use crate::circle::traits::IsCircleFriField;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Given g_n, a generator of the subgroup of size n of the circle, i.e. <g_n>,
/// and given a shift, that is a another point of the circle,
/// we define the coset shift + <g_n> which is the set of all the points in
/// <g_n> plus the shift.
/// For example, if <g_4> = {p1, p2, p3, p4}, then g_8 + <g_4> = {g_8 + p1, g_8 + p2, g_8 + p3, g_8 + p4}.

#[derive(Debug, Clone)]
pub struct Coset<F: IsCircleFriField> {
    // Coset: shift + <g_n> where n = 2^{log_2_size}.
    // Example: g_16 + <g_8>, n = 8, log_2_size = 3, shift = g_16.
    pub log_2_size: u32,
    pub shift: CirclePoint<F>,
}

impl<F: IsCircleFriField> Coset<F> {
    pub fn new(log_2_size: u32, shift: CirclePoint<F>) -> Self {
        Coset { log_2_size, shift }
    }

    /// Returns the coset g_2n + <g_n>
    pub fn new_standard(log_2_size: u32) -> Self {
        // shift is a generator of the subgroup of order 2n = 2^{log_2_size + 1}.
        let shift = CirclePoint::get_generator_of_subgroup(log_2_size + 1);
        Coset { log_2_size, shift }
    }

    /// Returns g_n, the generator of the subgroup of order n = 2^log_2_size.
    pub fn get_generator(&self) -> CirclePoint<F> {
        CirclePoint::GENERATOR.repeated_double(F::LOG_MAX_SUBGROUP_ORDER - self.log_2_size)
    }

    /// Given a standard coset g_2n + <g_n>, returns the subcoset with half size g_2n + <g_{n/2}>
    ///
    /// # Panics
    /// Panics if `log_2_size == 0` (cannot halve a coset of size 1).
    pub fn half_coset(&self) -> Self {
        assert!(
            self.log_2_size > 0,
            "Cannot halve a coset of size 1 (log_2_size == 0)"
        );
        Coset {
            log_2_size: self.log_2_size - 1,
            shift: self.shift.clone(),
        }
    }

    /// Given a coset shift + G returns the coset -shift + G.
    /// Note that (g_2n + <g_{n/2}>) U (-g_2n + <g_{n/2}>) = g_2n + <g_n>.
    pub fn conjugate(&self) -> Self {
        Coset {
            log_2_size: self.log_2_size,
            shift: self.shift.clone().conjugate(),
        }
    }

    /// Returns the vector of shift + g for every g in <g_n>.
    /// where g = i * g_n for i = 0, ..., n-1.
    #[cfg(feature = "alloc")]
    pub fn get_coset_points(&self) -> Vec<CirclePoint<F>> {
        // g_n the generator of the subgroup of order n.
        let generator_n = CirclePoint::get_generator_of_subgroup(self.log_2_size);
        let size: usize = 1 << self.log_2_size;
        core::iter::successors(Some(self.shift.clone()), move |prev| {
            Some(prev + &generator_n)
        })
        .take(size)
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::mersenne31::field::Mersenne31Field;

    type TestCoset = Coset<Mersenne31Field>;

    #[test]
    fn coset_points_vector_has_right_size() {
        let coset = TestCoset::new_standard(3);
        let points = coset.get_coset_points();
        assert_eq!(1 << coset.log_2_size, points.len())
    }

    #[test]
    fn antipode_of_coset_point_is_in_coset() {
        let coset = TestCoset::new_standard(3);
        let points = coset.get_coset_points();
        let point = points[2].clone();
        let antipode_point = points[6].clone();
        assert_eq!(antipode_point, point.antipode())
    }

    #[test]
    fn coset_generator_has_right_order() {
        let coset = TestCoset::new(2, CirclePoint::GENERATOR * 3);
        let generator_n = coset.get_generator();
        assert_eq!(generator_n.repeated_double(2), CirclePoint::zero());
    }

    #[test]
    #[should_panic(expected = "Cannot halve a coset of size 1")]
    fn half_coset_panics_on_size_one() {
        let coset = Coset::new(0, CirclePoint::GENERATOR);
        let _ = coset.half_coset(); // Should panic
    }
}
