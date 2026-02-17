/// Sampling functions for lattice-based cryptography.
///
/// Implements deterministic sampling from SHAKE-128/256 XOF streams,
/// following FIPS 204 (ML-DSA / Dilithium) algorithms.
///
/// All functions are deterministic: the same seed always produces the same output.
use alloc::vec;
use alloc::vec::Vec;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::dilithium_prime::DilithiumField;
use lambdaworks_math::polynomial::quotient_ring::PolynomialRingElement;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Shake128, Shake256};

type FE = FieldElement<DilithiumField>;

const Q: u64 = 8380417;

/// Generates the public matrix A ∈ Rq^{k×l} from a seed using SHAKE-128.
///
/// Each entry A[i][j] is a polynomial in Rq = Zq[X]/(X^N + 1) sampled
/// by rejection from a SHAKE-128 stream seeded with (seed || j || i).
///
/// Based on FIPS 204, Algorithm 30 (ExpandA).
pub fn expand_a<const N: usize>(
    seed: &[u8],
    k: usize,
    l: usize,
) -> Vec<Vec<PolynomialRingElement<DilithiumField, N>>> {
    let mut matrix = Vec::with_capacity(k);
    for i in 0..k {
        let mut row = Vec::with_capacity(l);
        for j in 0..l {
            let mut shake = Shake128::default();
            shake.update(seed);
            // Domain separation: append (j, i) as single bytes
            shake.update(&[j as u8, i as u8]);
            let mut reader = shake.finalize_xof();
            let poly = sample_ntt_poly::<N>(&mut reader);
            row.push(poly);
        }
        matrix.push(row);
    }
    matrix
}

/// Generates secret vectors with small coefficients in [-eta, eta].
///
/// Each polynomial is sampled from a SHAKE-256 stream seeded with
/// (seed || nonce), where nonce is the polynomial index.
///
/// Based on FIPS 204, Algorithm 31 (ExpandS).
pub fn expand_s<const N: usize>(
    seed: &[u8],
    eta: u32,
    count: usize,
) -> Vec<PolynomialRingElement<DilithiumField, N>> {
    let mut polys = Vec::with_capacity(count);
    for nonce in 0..count {
        let mut shake = Shake256::default();
        shake.update(seed);
        // 2-byte little-endian nonce
        shake.update(&(nonce as u16).to_le_bytes());
        let mut reader = shake.finalize_xof();
        let poly = sample_cbd::<N>(&mut reader, eta);
        polys.push(poly);
    }
    polys
}

/// Samples a single polynomial with uniform coefficients in [0, q) via rejection.
///
/// Reads 3 bytes at a time from the XOF stream, forms a 24-bit value,
/// and rejects values >= q.
///
/// Based on FIPS 204, Algorithm 14 (RejNTTPol).
fn sample_ntt_poly<const N: usize>(
    reader: &mut impl XofReader,
) -> PolynomialRingElement<DilithiumField, N> {
    let mut coeffs = Vec::with_capacity(N);
    while coeffs.len() < N {
        if let Some(val) = sample_uniform(reader) {
            coeffs.push(FE::from(val));
        }
    }
    PolynomialRingElement::new(&coeffs)
}

/// Rejection sampling: reads 3 bytes and returns a value in [0, q) or None.
///
/// Based on FIPS 204, Algorithm 14 (CoeffFromThreeBytes).
/// Reads 3 bytes, forms a 24-bit integer with the top bit of the third byte
/// masked off (giving a 23-bit value), and rejects if >= q.
pub fn sample_uniform(reader: &mut impl XofReader) -> Option<u64> {
    let mut buf = [0u8; 3];
    reader.read(&mut buf);
    let val = (buf[0] as u64) | ((buf[1] as u64) << 8) | (((buf[2] & 0x7F) as u64) << 16);
    if val < Q {
        Some(val)
    } else {
        None
    }
}

/// Samples a polynomial with coefficients in [-eta, eta] using centered binomial distribution.
///
/// For eta = 2: reads 1 byte per 2 coefficients.
/// For eta = 4: reads 1 byte per coefficient (4 bits each half).
///
/// Based on FIPS 204, Algorithm 15 (RejBoundedPoly / CBD).
fn sample_cbd<const N: usize>(
    reader: &mut impl XofReader,
    eta: u32,
) -> PolynomialRingElement<DilithiumField, N> {
    let mut coeffs = Vec::with_capacity(N);

    match eta {
        2 => {
            // For eta=2, read N/2 bytes (each byte gives 2 coefficients)
            let num_bytes = N / 2;
            let mut buf = vec![0u8; num_bytes];
            reader.read(&mut buf);

            for &byte in &buf {
                // First coefficient from low 4 bits
                let a0 = (byte & 1) + ((byte >> 1) & 1);
                let b0 = ((byte >> 2) & 1) + ((byte >> 3) & 1);
                coeffs.push(centered_to_fe(a0 as i32 - b0 as i32));

                // Second coefficient from high 4 bits
                let a1 = ((byte >> 4) & 1) + ((byte >> 5) & 1);
                let b1 = ((byte >> 6) & 1) + ((byte >> 7) & 1);
                coeffs.push(centered_to_fe(a1 as i32 - b1 as i32));
            }
        }
        4 => {
            // For eta=4, read N bytes (each byte gives 1 coefficient)
            let mut buf = vec![0u8; N];
            reader.read(&mut buf);

            for &byte in &buf {
                let a = (byte & 1) + ((byte >> 1) & 1) + ((byte >> 2) & 1) + ((byte >> 3) & 1);
                let b =
                    ((byte >> 4) & 1) + ((byte >> 5) & 1) + ((byte >> 6) & 1) + ((byte >> 7) & 1);
                coeffs.push(centered_to_fe(a as i32 - b as i32));
            }
        }
        _ => panic!("unsupported eta value: {eta} (expected 2 or 4)"),
    }

    coeffs.truncate(N);
    PolynomialRingElement::new(&coeffs)
}

/// Samples a mask polynomial y with coefficients in [-(gamma1-1), gamma1-1].
///
/// Reads bytes from a SHAKE-256 stream seeded with (seed || nonce).
/// For gamma1 = 2^17: 18-bit encoding (9 bytes per 4 coefficients).
/// For gamma1 = 2^19: 20-bit encoding (5 bytes per 2 coefficients).
///
/// Based on FIPS 204, Algorithm 34 (ExpandMask).
pub fn sample_mask<const N: usize>(
    seed: &[u8],
    gamma1: u32,
    nonce: u16,
) -> PolynomialRingElement<DilithiumField, N> {
    let mut shake = Shake256::default();
    shake.update(seed);
    shake.update(&nonce.to_le_bytes());
    let mut reader = shake.finalize_xof();

    let mut coeffs = Vec::with_capacity(N);

    match gamma1 {
        // gamma1 = 2^17 = 131072
        131072 => {
            // 18-bit encoding: 9 bytes → 4 coefficients
            while coeffs.len() < N {
                let mut buf = [0u8; 9];
                reader.read(&mut buf);

                // Extract four 18-bit values
                let r0 =
                    (buf[0] as u32) | ((buf[1] as u32) << 8) | (((buf[2] & 0x03) as u32) << 16);
                let r1 = ((buf[2] as u32) >> 2)
                    | ((buf[3] as u32) << 6)
                    | (((buf[4] & 0x0F) as u32) << 14);
                let r2 = ((buf[4] as u32) >> 4)
                    | ((buf[5] as u32) << 4)
                    | (((buf[6] & 0x3F) as u32) << 12);
                let r3 = ((buf[6] as u32) >> 6) | ((buf[7] as u32) << 2) | ((buf[8] as u32) << 10);

                for r in [r0, r1, r2, r3] {
                    if coeffs.len() < N {
                        // gamma1 - r gives centered representation
                        coeffs.push(centered_to_fe(gamma1 as i32 - r as i32));
                    }
                }
            }
        }
        // gamma1 = 2^19 = 524288
        524288 => {
            // 20-bit encoding: 5 bytes → 2 coefficients
            while coeffs.len() < N {
                let mut buf = [0u8; 5];
                reader.read(&mut buf);

                let r0 =
                    (buf[0] as u32) | ((buf[1] as u32) << 8) | (((buf[2] & 0x0F) as u32) << 16);
                let r1 = ((buf[2] as u32) >> 4) | ((buf[3] as u32) << 4) | ((buf[4] as u32) << 12);

                for r in [r0, r1] {
                    if coeffs.len() < N {
                        coeffs.push(centered_to_fe(gamma1 as i32 - r as i32));
                    }
                }
            }
        }
        _ => panic!("unsupported gamma1 value: {gamma1} (expected 2^17 or 2^19)"),
    }

    coeffs.truncate(N);
    PolynomialRingElement::new(&coeffs)
}

/// Samples the challenge polynomial c with exactly tau non-zero coefficients,
/// each being +1 or -1.
///
/// Based on FIPS 204, Algorithm 16 (SampleInBall).
/// The Hamming weight of the result is exactly tau.
pub fn sample_challenge<const N: usize>(
    seed: &[u8],
    tau: usize,
) -> PolynomialRingElement<DilithiumField, N> {
    let mut shake = Shake256::default();
    shake.update(seed);
    let mut reader = shake.finalize_xof();

    // Read 8 bytes to get the sign bits
    let mut sign_bytes = [0u8; 8];
    reader.read(&mut sign_bytes);
    let signs = u64::from_le_bytes(sign_bytes);

    let mut coeffs = vec![FE::from(0u64); N];

    for i in (N - tau)..N {
        // Sample j uniformly from [0, i]
        let j = sample_index(&mut reader, i + 1);

        // c[i] ← c[j], then c[j] ← ±1 (per FIPS 204 Algorithm 16)
        coeffs[i] = coeffs[j];

        let bit_index = i - (N - tau);
        let sign = (signs >> bit_index) & 1;
        if sign == 0 {
            coeffs[j] = FE::from(1u64);
        } else {
            coeffs[j] = FE::from(Q - 1); // -1 mod q
        }
    }

    PolynomialRingElement::new(&coeffs)
}

/// Samples a uniform index in [0, bound) by rejection from byte stream.
fn sample_index(reader: &mut impl XofReader, bound: usize) -> usize {
    loop {
        let mut buf = [0u8; 1];
        reader.read(&mut buf);
        let val = buf[0] as usize;
        if val < bound {
            return val;
        }
        // For bounds > 256, read more bytes
        if bound > 256 {
            let mut buf2 = [0u8; 2];
            reader.read(&mut buf2);
            let val = u16::from_le_bytes(buf2) as usize;
            if val < bound {
                return val;
            }
        }
    }
}

/// Converts a centered integer value to a field element.
/// Negative values are mapped to their modular equivalent: v + q.
fn centered_to_fe(v: i32) -> FE {
    if v >= 0 {
        FE::from(v as u64)
    } else {
        FE::from((Q as i64 + v as i64) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::polynomial::quotient_ring::centered_mod;

    #[test]
    fn expand_a_deterministic() {
        let seed = [42u8; 32];
        let a1 = expand_a::<256>(&seed, 4, 4);
        let a2 = expand_a::<256>(&seed, 4, 4);
        assert_eq!(a1.len(), 4);
        assert_eq!(a1[0].len(), 4);
        // Same seed produces same result
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(a1[i][j], a2[i][j]);
            }
        }
    }

    #[test]
    fn expand_a_different_seeds_differ() {
        let seed1 = [1u8; 32];
        let seed2 = [2u8; 32];
        let a1 = expand_a::<256>(&seed1, 2, 2);
        let a2 = expand_a::<256>(&seed2, 2, 2);
        // Different seeds should (overwhelmingly likely) give different results
        assert_ne!(a1[0][0], a2[0][0]);
    }

    #[test]
    fn expand_a_coefficients_in_range() {
        let seed = [0u8; 32];
        let a = expand_a::<256>(&seed, 2, 2);
        for row in &a {
            for poly in row {
                for c in poly.coefficients() {
                    let val: u64 = c.canonical();
                    assert!(val < Q, "coefficient {val} is not in [0, q)");
                }
            }
        }
    }

    #[test]
    fn sample_uniform_values_in_range() {
        let mut shake = Shake128::default();
        shake.update(b"test seed");
        let mut reader = shake.finalize_xof();

        let mut count = 0;
        for _ in 0..10000 {
            if let Some(val) = sample_uniform(&mut reader) {
                assert!(val < Q);
                count += 1;
            }
        }
        // With 23-bit values and q ≈ 2^23, acceptance rate is ~100%
        // We should get most values accepted
        assert!(
            count > 9000,
            "too many rejections: only {count}/10000 accepted"
        );
    }

    #[test]
    fn expand_s_eta_2_range() {
        let seed = [7u8; 32];
        let polys = expand_s::<256>(&seed, 2, 4);
        assert_eq!(polys.len(), 4);

        for poly in &polys {
            for c in poly.coefficients() {
                let centered = centered_mod::<DilithiumField>(c);
                assert!(
                    (-2..=2).contains(&centered),
                    "coefficient {centered} not in [-2, 2]"
                );
            }
        }
    }

    #[test]
    fn expand_s_eta_4_range() {
        let seed = [11u8; 32];
        let polys = expand_s::<256>(&seed, 4, 4);
        assert_eq!(polys.len(), 4);

        for poly in &polys {
            for c in poly.coefficients() {
                let centered = centered_mod::<DilithiumField>(c);
                assert!(
                    (-4..=4).contains(&centered),
                    "coefficient {centered} not in [-4, 4]"
                );
            }
        }
    }

    #[test]
    fn expand_s_deterministic() {
        let seed = [42u8; 64];
        let s1 = expand_s::<256>(&seed, 2, 4);
        let s2 = expand_s::<256>(&seed, 2, 4);
        for i in 0..4 {
            assert_eq!(s1[i], s2[i]);
        }
    }

    #[test]
    fn sample_challenge_hamming_weight() {
        let seed = [99u8; 32];
        let tau = 39; // Dilithium-II tau
        let c = sample_challenge::<256>(&seed, tau);

        // Count non-zero coefficients
        let padded = c.padded_coefficients();
        let nonzero: usize = padded.iter().filter(|c| **c != FE::from(0u64)).count();
        assert_eq!(nonzero, tau, "Hamming weight should be exactly tau={tau}");
    }

    #[test]
    fn sample_challenge_values_are_plus_minus_one() {
        let seed = [55u8; 32];
        let tau = 39;
        let c = sample_challenge::<256>(&seed, tau);

        let padded = c.padded_coefficients();
        for coeff in &padded {
            let centered = centered_mod::<DilithiumField>(coeff);
            assert!(
                centered == 0 || centered == 1 || centered == -1,
                "challenge coefficient must be 0, +1, or -1, got {centered}"
            );
        }
    }

    #[test]
    fn sample_challenge_deterministic() {
        let seed = [77u8; 32];
        let c1 = sample_challenge::<256>(&seed, 39);
        let c2 = sample_challenge::<256>(&seed, 39);
        assert_eq!(c1, c2);
    }

    #[test]
    fn sample_mask_gamma1_2_17_range() {
        let seed = [13u8; 64];
        let gamma1: u32 = 1 << 17; // 131072
        let y = sample_mask::<256>(&seed, gamma1, 0);

        for c in y.coefficients() {
            let centered = centered_mod::<DilithiumField>(c);
            let bound = (gamma1 - 1) as i64;
            assert!(
                centered.abs() <= bound,
                "mask coefficient {centered} not in [-{bound}, {bound}]"
            );
        }
    }

    #[test]
    fn sample_mask_gamma1_2_19_range() {
        let seed = [17u8; 64];
        let gamma1: u32 = 1 << 19; // 524288
        let y = sample_mask::<256>(&seed, gamma1, 0);

        for c in y.coefficients() {
            let centered = centered_mod::<DilithiumField>(c);
            let bound = (gamma1 - 1) as i64;
            assert!(
                centered.abs() <= bound,
                "mask coefficient {centered} not in [-{bound}, {bound}]"
            );
        }
    }

    #[test]
    fn sample_mask_deterministic() {
        let seed = [21u8; 64];
        let y1 = sample_mask::<256>(&seed, 1 << 17, 0);
        let y2 = sample_mask::<256>(&seed, 1 << 17, 0);
        assert_eq!(y1, y2);
    }

    #[test]
    fn sample_mask_different_nonces_differ() {
        let seed = [21u8; 64];
        let y0 = sample_mask::<256>(&seed, 1 << 17, 0);
        let y1 = sample_mask::<256>(&seed, 1 << 17, 1);
        assert_ne!(y0, y1);
    }

    #[test]
    fn centered_to_fe_roundtrip() {
        // Positive value
        let fe_pos = centered_to_fe(42);
        assert_eq!(centered_mod::<DilithiumField>(&fe_pos), 42);

        // Negative value
        let fe_neg = centered_to_fe(-42);
        assert_eq!(centered_mod::<DilithiumField>(&fe_neg), -42);

        // Zero
        let fe_zero = centered_to_fe(0);
        assert_eq!(centered_mod::<DilithiumField>(&fe_zero), 0);
    }
}
