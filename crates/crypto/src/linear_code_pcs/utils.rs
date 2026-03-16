/// Number of column openings needed for `sec_param` bits of security with
/// code relative distance `delta_num / delta_den`.
///
/// `t = ceil(sec_param / log2(1 / (1 - delta)))`, capped at `n_ext_cols`
/// so we never open more columns than exist.
pub fn calculate_t(
    sec_param: usize,
    delta_num: usize,
    delta_den: usize,
    n_ext_cols: usize,
) -> usize {
    let delta = (delta_num as f64) / (delta_den as f64);
    // log2(1 / (1 - delta))
    let log_factor = -(1.0 - delta).log2();
    let t = (sec_param as f64 / log_factor).ceil() as usize;
    t.min(n_ext_cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_t_rs_rate_half() {
        // RS rate 1/2: delta = 1/2. Formula: t = ceil(128 / log2(1/(1-0.5))) = ceil(128/1) = 128
        let t = calculate_t(128, 1, 2, 1000);
        assert_eq!(t, 128);
    }

    #[test]
    fn calculate_t_rs_rate_quarter() {
        // RS rate 1/4: delta = 3/4. Formula: t = ceil(128 / log2(1/(1-0.75))) = ceil(128/2) = 64
        let t = calculate_t(128, 3, 4, 1000);
        assert_eq!(t, 64);
    }

    #[test]
    fn calculate_t_capped_at_n_ext_cols() {
        let t = calculate_t(128, 1, 2, 50);
        assert_eq!(t, 50);
    }

    #[test]
    fn calculate_t_small_distance() {
        // delta = 1/25 = 0.04
        let t = calculate_t(128, 1, 25, 10000);
        let expected = (128.0_f64 / (-(1.0 - 1.0 / 25.0_f64).log2())).ceil() as usize;
        assert_eq!(t, expected);
    }
}
