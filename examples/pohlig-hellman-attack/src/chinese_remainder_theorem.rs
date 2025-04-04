#[derive(Debug)]
pub enum ChineseRemainderTheoremError {
    ModuliNotCoprime,
}

/// Given an array [(x_1, n_1), ... , (x_k, n_k)] it returns x such that
/// x ≡ x_1 mod n_1,
/// x ≡ x_2 mod n_2,
/// ...
/// x ≡ x_k mod n_k.
///
/// The moduli n_i should be pairwise coprimes.
/// The Chinese Remainder Theorem states that if the moduli n_i are pairwise coprimes,
/// then there is always a unique solution x to these equations in the range [0, N-1]
/// where N is the product of all moduli.
pub fn chinese_remainder_theorem(
    equations: &[(i128, i128)],
) -> Result<i128, ChineseRemainderTheoremError> {
    // Calculate the product of all moduli
    let n: i128 = equations.iter().map(|&(_, m)| m).product();

    // For each equation, compute:
    // 1. n_i = n / m_i (product of all moduli except current)
    // 2. x_i = inverse of n_i modulo m_i.
    // 3. Add a_i * n_i * x_i to the result.
    let mut result = 0;
    for &(a, m) in equations {
        let n_i = n / m;

        // Find x_i such that n_i * x_i ≡ 1 (mod m)
        let x_i = mod_inverse(n_i, m).ok_or(ChineseRemainderTheoremError::ModuliNotCoprime)?;

        // Compute (result + a * n_i * x_i) % n
        result = (result + a * n_i * x_i) % n;
    }

    Ok((result % n + n) % n)
}

/// Computes the modular multiplicative inverse of x modulo n.
///
/// If gcd(x, n) = 1, then there exists a value y such that (x * y) % n = 1.
/// This y is the modular multiplicative inverse of x.
///
/// # Returns
/// * `Some(y)` if the inverse exists
/// * `None` if x and n are not coprime (gcd(x, n) ≠ 1)
pub fn mod_inverse(x: i128, n: i128) -> Option<i128> {
    let (g, x, _) = extended_euclidean_algorithm(x, n);
    if g == 1 {
        Some((x % n + n) % n)
    } else {
        None
    }
}

/// Computes the extended Euclidean algorithm for two integers.
///
/// Given a and b, it returns (g, s, t) where:
/// g = gcd(a, b)
/// s and t are Bézout coefficients such that s*a + t*b = g
///
/// # Arguments
/// * `a` - First integer
/// * `b` - Second integer
///
/// # Returns
/// * A tuple (g, s, t) as described above
pub fn extended_euclidean_algorithm(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut s, mut old_s) = (0, 1);
    let (mut t, mut old_t) = (1, 0);
    let (mut r, mut old_r) = (b, a);

    while r != 0 {
        let quotient = old_r / r;

        // Update remainders
        let temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        // Update coefficients
        let temp_s = s;
        s = old_s - quotient * s;
        old_s = temp_s;

        let temp_t = t;
        t = old_t - quotient * t;
        old_t = temp_t;
    }

    (old_r, old_s, old_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 7), Some(5));
        assert_eq!(mod_inverse(17, 3120), Some(2753));
        assert_eq!(mod_inverse(2, 4), None); // gcd(2, 4) = 2, so no inverse
        assert_eq!(mod_inverse(1, 5), Some(1));
    }

    #[test]
    fn test_chinese_remainder_theorem() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
        let equations = [(2, 3), (3, 5), (2, 7)];
        assert_eq!(chinese_remainder_theorem(&equations).unwrap(), 23);

        // x ≡ 1 (mod 4), x ≡ 2 (mod 27)
        let equations = [(1, 4), (2, 27)];
        assert_eq!(chinese_remainder_theorem(&equations).unwrap(), 29);

        // x ≡ 1 (mod 3), x ≡ 2 (mod 4), x ≡ 3 (mod 5)
        let equations = [(1, 3), (2, 4), (3, 5)];
        assert_eq!(chinese_remainder_theorem(&equations).unwrap(), 58);
    }
}
