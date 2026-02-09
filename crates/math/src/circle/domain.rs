extern crate alloc;
use crate::circle::cosets::Coset;
use crate::circle::point::CirclePoint;
use crate::circle::traits::IsCircleFriField;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// A circle domain is a coset of a circle subgroup, used as an evaluation domain.
/// It wraps `Coset` and adds domain-level operations needed for Circle FRI.
#[derive(Debug, Clone)]
pub struct CircleDomain<F: IsCircleFriField> {
    pub coset: Coset<F>,
}

impl<F: IsCircleFriField> CircleDomain<F> {
    /// Creates a standard domain of size 2^log_2_size.
    /// This is the coset g_{2n} + <g_n>.
    pub fn new_standard(log_2_size: u32) -> Self {
        Self {
            coset: Coset::new_standard(log_2_size),
        }
    }

    pub fn new(coset: Coset<F>) -> Self {
        Self { coset }
    }

    pub fn log_2_size(&self) -> u32 {
        self.coset.log_2_size
    }

    pub fn size(&self) -> usize {
        1 << self.coset.log_2_size
    }

    /// Returns all points in the domain.
    #[cfg(feature = "alloc")]
    pub fn get_points(&self) -> Vec<CirclePoint<F>> {
        Coset::get_coset_points(&self.coset)
    }

    /// After the first fold (y-fold), the domain projects to x-coordinates.
    /// The resulting domain for subsequent folds is the half-coset with the same shift.
    pub fn fold_y(&self) -> CircleDomain<F> {
        CircleDomain {
            coset: Coset::half_coset(self.coset.clone()),
        }
    }

    /// After an x-fold, the domain maps via x -> t = 2x^2 - 1.
    /// The resulting domain is a standard coset of half the current size.
    pub fn fold_x(&self) -> CircleDomain<F> {
        CircleDomain {
            coset: Coset::new_standard(self.coset.log_2_size - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::mersenne31::field::Mersenne31Field;

    type TestDomain = CircleDomain<Mersenne31Field>;

    #[test]
    fn standard_domain_has_correct_size() {
        let domain = TestDomain::new_standard(4);
        assert_eq!(domain.size(), 16);
    }

    #[test]
    fn fold_y_halves_domain() {
        let domain = TestDomain::new_standard(4);
        let folded = domain.fold_y();
        assert_eq!(folded.size(), 8);
    }

    #[test]
    fn fold_x_halves_domain_again() {
        let domain = TestDomain::new_standard(4);
        let folded = domain.fold_y().fold_x();
        assert_eq!(folded.size(), 4);
    }

    #[test]
    fn repeated_folds_reduce_to_one() {
        let domain = TestDomain::new_standard(3); // 8 points
        let d1 = domain.fold_y(); // 4 points
        assert_eq!(d1.size(), 4);
        let d2 = d1.fold_x(); // 2 points
        assert_eq!(d2.size(), 2);
        let d3 = d2.fold_x(); // 1 point
        assert_eq!(d3.size(), 1);
    }

    #[test]
    fn standard_coset_conjugate_pairing() {
        // In a standard coset of size n, the conjugate of point[i] is point[n-1-i].
        // Conjugate pairs share the same x-coordinate and have opposite y-coordinates.
        let domain = TestDomain::new_standard(3);
        let points = domain.get_points();
        let n = points.len();

        for i in 0..n / 2 {
            assert_eq!(
                points[i].x,
                points[n - 1 - i].x,
                "Points {} and {} should have the same x-coordinate",
                i,
                n - 1 - i
            );
            assert_eq!(
                points[i].y,
                -points[n - 1 - i].y.clone(),
                "Points {} and {} should have opposite y-coordinates",
                i,
                n - 1 - i
            );
        }
    }

    #[test]
    fn standard_coset_conjugate_pairing_16() {
        let domain = TestDomain::new_standard(4);
        let points = domain.get_points();
        let n = points.len();

        for i in 0..n / 2 {
            assert_eq!(points[i].x, points[n - 1 - i].x);
            assert_eq!(points[i].y, -points[n - 1 - i].y.clone());
        }
    }

    #[test]
    fn standard_coset_antipode_pairing() {
        // In a standard coset of size n, point[i] and point[i+n/2] are antipodes:
        // (-x, -y) relationship. This is used for x-fold pairing.
        let domain = TestDomain::new_standard(3);
        let points = domain.get_points();
        let n = points.len();
        let half = n / 2;

        for i in 0..half {
            assert_eq!(
                points[i].x,
                -points[i + half].x.clone(),
                "Points {} and {} should have opposite x-coordinates (antipode)",
                i,
                i + half
            );
            assert_eq!(
                points[i].y,
                -points[i + half].y.clone(),
                "Points {} and {} should have opposite y-coordinates (antipode)",
                i,
                i + half
            );
        }
    }
}
