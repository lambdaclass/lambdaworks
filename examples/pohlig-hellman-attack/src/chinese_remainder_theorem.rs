/// Given an array [(x_1, n_1), ... , (x_k, n_k)] it returns x such that
/// x ≡ x_1 mod n_1
/// x ≡ x_2 mod n_2
/// ...
/// x ≡ x_k mod n_k
/// The moduli n_i should be pairwise coprimes.
/// The Chinese Remainder Theorem states that if the moduli n_i are pairwise coprimes, then there is always a solution x.
pub fn chinese_remainder_theorem(equations: &[(i128, i128)]) -> i128 {
    // Calculate the product of all moduli
    let n: i128 = equations.iter().map(|(_, m)| m).product();

    // For each equation, compute:
    // 1. n_i = n / m_i (product of all moduli except current)
    // 2. x_i = inverse of n_i modulo m_i
    // 3. Add a_i * n_i * x_i to the result
    let mut result = 0;
    for &(a, m) in equations {
        let n_i = n / m;
        // Find x_i such that n_i * x_i ≡ 1 (mod m)
        let x_i = mod_inverse(n_i, m).expect("Moduli must be pairwise coprime");
        result = (result + a * n_i * x_i) % n;
    }

    result
}

// returns the inverse of x modulus n.
// If gcd(x, n) = 1, then sx + tn = 1 means that sx = 1 mod n. Then, s is the inverse of x mod n.
// However, the x value returned by the Extended Euclidean Algorithm could be negative or larger than n.
// In modular arithmetic, we typically want the representative in the range [0, n-1].
// The expression (x % n + n) % n handles both cases.
// For example, if n = 7 and the algorithm gives x = -3:
//  -3 % 7 = -3 (in Rust)
//  -3 + 7 = 4
//  4 % 7 = 4
//  So 4 is the modular inverse, which means (a * 4) % 7 = 1.
pub fn mod_inverse(x: i128, n: i128) -> Option<i128> {
    let (g, x, _) = extended_euclidean_algorithm(x, n);
    if g == 1 {
        Some((x % n + n) % n)
    } else {
        None
    }
}

// given a and b it returns (g, s, t) where
// g = the greatest common divisor between a and b = gcd(a, b)
// s and t the bezout coefficients such that sa + tb = g
pub fn extended_euclidean_algorithm(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut rem) = (a, b);
    let (mut old_s, mut coeff_s) = (1, 0);
    let (mut old_t, mut coeff_t) = (0, 1);

    while rem != 0 {
        let quotient = old_r / rem;

        update_step(&mut rem, &mut old_r, quotient);
        update_step(&mut coeff_s, &mut old_s, quotient);
        update_step(&mut coeff_t, &mut old_t, quotient);
    }

    (old_r, old_s, old_t)
}

pub fn update_step(a: &mut i128, old_a: &mut i128, quotient: i128) {
    let temp = *a;
    *a = *old_a - quotient * temp;
    *old_a = temp;
}
