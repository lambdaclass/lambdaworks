/// Number of column openings needed for `sec_param` bits of security with
/// code relative distance `delta_num / delta_den`.
///
/// From Brakedown (Golovnev et al., CRYPTO 2023), Lemma 1 (p.14):
/// The soundness error of the testing phase is `(1 - δ/3)^t`, where δ is the
/// code's relative minimum distance and t is the number of opened columns.
/// For λ bits of security: `t = ceil(λ / log2(1 / (1 - δ/3)))`.
///
/// Capped at `n_ext_cols` so we never open more columns than exist.
pub fn calculate_t(
    sec_param: usize,
    delta_num: usize,
    delta_den: usize,
    n_ext_cols: usize,
) -> usize {
    // δ/3: the per-column catch probability from the Brakedown soundness analysis.
    // The factor 1/3 comes from Claim 1 + Lemma 1: at most (δ/3)·N columns can be
    // "bad" (where committed rows differ from their closest codewords), so each
    // random column catches the cheat with probability ≥ δ/3.
    let delta_third = (delta_num as f64) / (delta_den as f64) / 3.0;
    let log_factor = -libm::log2(1.0 - delta_third);
    let t = libm::ceil(sec_param as f64 / log_factor) as usize;
    t.min(n_ext_cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_t_rs_rate_half() {
        // RS rate 1/2: δ = 1/2, δ/3 = 1/6
        // t = ceil(128 / log2(1/(1 - 1/6))) = ceil(128 / log2(6/5)) = ceil(128 / 0.2630) = 487
        let t = calculate_t(128, 1, 2, 1000);
        let expected = libm::ceil(128.0 / (-libm::log2(1.0 - 1.0 / 6.0))) as usize;
        assert_eq!(t, expected);
    }

    #[test]
    fn calculate_t_rs_rate_quarter() {
        // RS rate 1/4: δ = 3/4, δ/3 = 1/4
        // t = ceil(128 / log2(1/(1 - 1/4))) = ceil(128 / log2(4/3)) = ceil(128 / 0.4150) = 309
        let t = calculate_t(128, 3, 4, 1000);
        let expected = libm::ceil(128.0 / (-libm::log2(1.0 - 0.25))) as usize;
        assert_eq!(t, expected);
    }

    #[test]
    fn calculate_t_capped_at_n_ext_cols() {
        let t = calculate_t(128, 1, 2, 50);
        assert_eq!(t, 50);
    }

    #[test]
    fn calculate_t_small_distance() {
        // δ = 1/25 = 0.04, δ/3 ≈ 0.01333
        let t = calculate_t(128, 1, 25, 100000);
        let expected = libm::ceil(128.0_f64 / (-libm::log2(1.0 - 1.0 / 75.0_f64))) as usize;
        assert_eq!(t, expected);
    }
}
