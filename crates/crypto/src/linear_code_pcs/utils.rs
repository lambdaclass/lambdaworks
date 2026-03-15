/// Number of column openings needed for `sec_param` bits of security with
/// code relative distance `delta_num / delta_den`.
///
/// `t = ceil(sec_param / log2(1 / (1 - delta/2)))`, capped at `n_ext_cols`
/// so we never open more columns than exist.
pub fn calculate_t(
    sec_param: usize,
    delta_num: usize,
    delta_den: usize,
    n_ext_cols: usize,
) -> usize {
    // delta/2
    let half_delta = (delta_num as f64) / (delta_den as f64) / 2.0;
    // log2(1 / (1 - delta/2))
    let log_factor = -(1.0 - half_delta).log2();
    let t = (sec_param as f64 / log_factor).ceil() as usize;
    t.min(n_ext_cols)
}
