use super::point::HasCircleParams;
use crate::field::traits::IsField;

/// Marker trait for fields that support Circle FRI with 2-power subgroups.
///
/// The circle group over such a field has a subgroup of order `2^LOG_MAX_SUBGROUP_ORDER`,
/// analogous to `IsFFTField` for multiplicative FFTs.
pub trait IsCircleFriField: IsField + HasCircleParams<Self> {
    /// The log₂ of the maximum circle subgroup order.
    /// For Mersenne31 this is 31 (the circle group has order 2^31).
    const LOG_MAX_SUBGROUP_ORDER: u32;
}
