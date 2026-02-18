//! # Ring-LWE encryption
//!
//! Educational implementation of Ring-LWE public-key encryption over the
//! polynomial ring Rq = Zq[X]/(X^N + 1), using lambdaworks' DilithiumField
//! (q = 8380417, N = 256).
//!
//! Shows how replacing LWE's matrix A with a single ring element gives
//! the same security with dramatically smaller keys.
//!
//! **Not cryptographically secure** — intended as an educational example.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::dilithium_prime::DilithiumField;
use lambdaworks_math::polynomial::quotient_ring::PolynomialRingElement;
use rand::Rng;

pub const Q: u64 = 8380417;
pub const N: usize = 256;

type FE = FieldElement<DilithiumField>;
type Rq = PolynomialRingElement<DilithiumField, N>;

/// Ring-LWE public key: (a, b) where b = a·s + e.
pub struct PublicKey {
    pub a: Rq,
    pub b: Rq,
}

/// Ring-LWE secret key: the small secret polynomial s.
pub struct SecretKey {
    pub s: Rq,
}

/// Ring-LWE ciphertext: (u, v).
pub struct Ciphertext {
    pub u: Rq,
    pub v: Rq,
}

/// Generates a random ring element with all N coefficients uniform in [0, q).
fn random_ring_element<R: Rng>(rng: &mut R) -> Rq {
    let coeffs: Vec<FE> = (0..N).map(|_| FE::from(rng.gen_range(0..Q))).collect();
    Rq::new(&coeffs)
}

/// Generates a small ring element with coefficients in [-bound, bound].
fn small_ring_element<R: Rng>(rng: &mut R, bound: u64) -> Rq {
    let coeffs: Vec<FE> = (0..N)
        .map(|_| {
            let val = rng.gen_range(0..=2 * bound) as i64 - bound as i64;
            if val >= 0 {
                FE::from(val as u64)
            } else {
                -FE::from((-val) as u64)
            }
        })
        .collect();
    Rq::new(&coeffs)
}

/// Generates a Ring-LWE key pair.
///
/// - `error_bound`: coefficients of secret and error polynomials are in [-error_bound, error_bound]
pub fn keygen<R: Rng>(rng: &mut R, error_bound: u64) -> (PublicKey, SecretKey) {
    let a = random_ring_element(rng);
    let s = small_ring_element(rng, error_bound);
    let e = small_ring_element(rng, error_bound);

    // b = a·s + e
    let b = &a.mul_ntt(&s) + &e;

    (PublicKey { a, b }, SecretKey { s })
}

/// Encrypts a single bit (0 or 1).
///
/// Chooses small random polynomials r, e1, e2 and computes:
/// - u = a·r + e1
/// - v = b·r + e2 + bit·⌊q/2⌋
pub fn encrypt<R: Rng>(rng: &mut R, pk: &PublicKey, bit: u8) -> Ciphertext {
    assert!(bit <= 1, "Can only encrypt a single bit (0 or 1)");

    let r = small_ring_element(rng, 1);
    let e1 = small_ring_element(rng, 1);
    let e2 = small_ring_element(rng, 1);

    let u = &pk.a.mul_ntt(&r) + &e1;

    // Encode the bit as ⌊q/2⌋ in the constant coefficient
    let mut msg_coeffs = vec![FE::from(0u64); N];
    msg_coeffs[0] = FE::from(bit as u64 * (Q / 2));
    let msg = Rq::new(&msg_coeffs);

    let v = &(&pk.b.mul_ntt(&r) + &e2) + &msg;

    Ciphertext { u, v }
}

/// Decrypts a ciphertext to recover the encrypted bit.
///
/// Computes d = v - s·u. The constant coefficient of d should be
/// close to 0 (bit = 0) or ⌊q/2⌋ (bit = 1).
pub fn decrypt(sk: &SecretKey, ct: &Ciphertext) -> u8 {
    let su = sk.s.mul_ntt(&ct.u);
    let d = &ct.v - &su;

    // Check the constant coefficient
    let coeff = d.coefficient(0);
    let val = to_u64(&coeff);

    let half = Q / 2;
    let dist_to_zero = val.min(Q - val);
    let dist_to_half = val.abs_diff(half);

    if dist_to_half < dist_to_zero {
        1
    } else {
        0
    }
}

/// Extracts the canonical representative in [0, q) from a field element.
fn to_u64(x: &FE) -> u64 {
    x.canonical()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn make_rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(12345)
    }

    #[test]
    fn encrypt_decrypt_zero() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 2);
        let ct = encrypt(&mut rng, &pk, 0);
        assert_eq!(decrypt(&sk, &ct), 0);
    }

    #[test]
    fn encrypt_decrypt_one() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 2);
        let ct = encrypt(&mut rng, &pk, 1);
        assert_eq!(decrypt(&sk, &ct), 1);
    }

    #[test]
    fn encrypt_decrypt_multiple_bits() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 2);

        for expected_bit in [0u8, 1, 1, 0, 1, 0, 0, 1, 1, 0] {
            let ct = encrypt(&mut rng, &pk, expected_bit);
            assert_eq!(decrypt(&sk, &ct), expected_bit);
        }
    }

    #[test]
    fn key_sizes_are_compact() {
        // Ring-LWE public key is just 2 polynomials = 2 × 256 coefficients = 512 field elements
        // Equivalent LWE with same security would need an m×n matrix (thousands of elements)
        let mut rng = make_rng();
        let (pk, _sk) = keygen(&mut rng, 2);

        let pk_coeffs = pk.a.padded_coefficients().len() + pk.b.padded_coefficients().len();
        assert_eq!(pk_coeffs, 2 * N); // 512 coefficients total
    }

    #[test]
    fn ring_structure_wraps() {
        // Demonstrate X^N ≡ -1 in the ring
        let mut coeffs = vec![FE::from(0u64); N + 1];
        coeffs[N] = FE::from(1u64); // X^256
        let wrapped = Rq::new(&coeffs);

        // X^256 ≡ -1 mod (X^256 + 1), so constant term is q - 1
        assert_eq!(wrapped.coefficient(0), FE::from(Q - 1));
    }
}
