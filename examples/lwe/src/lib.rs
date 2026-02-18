//! # Learning With Errors (LWE) encryption
//!
//! Educational implementation of Regev's LWE-based public-key encryption scheme.
//! Encrypts single bits using the hardness of the LWE problem.
//!
//! **Not cryptographically secure** — uses toy parameters for readability.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use rand::Rng;

/// Small prime for toy LWE. Large enough for the scheme to work,
/// small enough that you can verify the math by hand.
pub const Q: u64 = 97;

type F = U64PrimeField<Q>;
type FE = FieldElement<F>;

/// LWE public key: matrix A and vector b = A·s + e.
pub struct PublicKey {
    /// Random matrix A ∈ Zq^{m×n}
    pub a: Vec<Vec<FE>>,
    /// b = A·s + e
    pub b: Vec<FE>,
}

/// LWE secret key: the secret vector s ∈ Zq^n.
pub struct SecretKey {
    pub s: Vec<FE>,
}

/// LWE ciphertext: (u, v) where u ∈ Zq^n and v ∈ Zq.
pub struct Ciphertext {
    pub u: Vec<FE>,
    pub v: FE,
}

/// Generates an LWE key pair.
///
/// - `n`: secret dimension (e.g. 4)
/// - `m`: number of samples / rows of A (e.g. 8)
/// - `error_bound`: coefficients of error vector are in [-error_bound, error_bound]
pub fn keygen<R: Rng>(rng: &mut R, n: usize, m: usize, error_bound: u64) -> (PublicKey, SecretKey) {
    // Secret vector s ∈ Zq^n
    let s: Vec<FE> = (0..n).map(|_| FE::from(rng.gen_range(0..Q))).collect();

    // Random matrix A ∈ Zq^{m×n}
    let a: Vec<Vec<FE>> = (0..m)
        .map(|_| (0..n).map(|_| FE::from(rng.gen_range(0..Q))).collect())
        .collect();

    // Small error vector e ∈ Zq^m
    let e: Vec<FE> = (0..m)
        .map(|_| {
            let err = rng.gen_range(0..=2 * error_bound) as i64 - error_bound as i64;
            if err >= 0 {
                FE::from(err as u64)
            } else {
                -FE::from((-err) as u64)
            }
        })
        .collect();

    // b = A·s + e
    let b: Vec<FE> = a
        .iter()
        .zip(e.iter())
        .map(|(row, ei)| {
            let dot: FE = row.iter().zip(s.iter()).map(|(aij, sj)| aij * sj).sum();
            dot + ei
        })
        .collect();

    (PublicKey { a, b }, SecretKey { s })
}

/// Encrypts a single bit (0 or 1).
///
/// Chooses a random binary vector r ∈ {0,1}^m (subset selection),
/// then computes u = A^T·r and v = b^T·r + bit·⌊q/2⌋.
pub fn encrypt<R: Rng>(rng: &mut R, pk: &PublicKey, bit: u8) -> Ciphertext {
    assert!(bit <= 1, "Can only encrypt a single bit (0 or 1)");

    let m = pk.a.len();
    let n = pk.a[0].len();

    // Random binary vector r ∈ {0,1}^m
    let r: Vec<u64> = (0..m).map(|_| rng.gen_range(0..=1u64)).collect();

    // u = A^T · r (vector in Zq^n)
    let u: Vec<FE> = (0..n)
        .map(|j| (0..m).map(|i| pk.a[i][j] * FE::from(r[i])).sum::<FE>())
        .collect();

    // v = b^T · r + bit * ⌊q/2⌋
    let br: FE =
        pk.b.iter()
            .zip(r.iter())
            .map(|(bi, ri)| bi * FE::from(*ri))
            .sum();
    let msg = FE::from(bit as u64 * (Q / 2));
    let v = br + msg;

    Ciphertext { u, v }
}

/// Decrypts a ciphertext to recover the encrypted bit.
///
/// Computes d = v - s^T·u, then checks whether d is closer
/// to 0 (bit = 0) or to ⌊q/2⌋ (bit = 1).
pub fn decrypt(sk: &SecretKey, ct: &Ciphertext) -> u8 {
    // d = v - s^T · u
    let su: FE = sk.s.iter().zip(ct.u.iter()).map(|(si, ui)| si * ui).sum();
    let d = ct.v - su;

    // Extract the representative in [0, q)
    let val = to_u64(&d);

    // If closer to q/2 than to 0, the bit is 1
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
        ChaCha20Rng::seed_from_u64(42)
    }

    #[test]
    fn encrypt_decrypt_zero() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 4, 8, 1);
        let ct = encrypt(&mut rng, &pk, 0);
        assert_eq!(decrypt(&sk, &ct), 0);
    }

    #[test]
    fn encrypt_decrypt_one() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 4, 8, 1);
        let ct = encrypt(&mut rng, &pk, 1);
        assert_eq!(decrypt(&sk, &ct), 1);
    }

    #[test]
    fn encrypt_decrypt_multiple_bits() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 4, 8, 1);

        for expected_bit in [0u8, 1, 1, 0, 1, 0, 0, 1] {
            let ct = encrypt(&mut rng, &pk, expected_bit);
            assert_eq!(decrypt(&sk, &ct), expected_bit);
        }
    }

    #[test]
    fn larger_parameters() {
        let mut rng = make_rng();
        let (pk, sk) = keygen(&mut rng, 8, 20, 1);

        for _ in 0..20 {
            let bit = rng.gen_range(0..=1u8);
            let ct = encrypt(&mut rng, &pk, bit);
            assert_eq!(decrypt(&sk, &ct), bit);
        }
    }

    #[test]
    fn wrong_key_decrypts_incorrectly() {
        let mut rng = make_rng();
        let (pk, _sk) = keygen(&mut rng, 4, 8, 1);
        let (_pk2, sk2) = keygen(&mut rng, 4, 8, 1);

        // Encrypt several bits and count how many decrypt correctly with wrong key
        let mut correct = 0;
        let total = 20;
        for _ in 0..total {
            let bit = 1u8;
            let ct = encrypt(&mut rng, &pk, bit);
            if decrypt(&sk2, &ct) == bit {
                correct += 1;
            }
        }

        // With wrong key, decryption should fail most of the time
        // (probabilistically — about half will be wrong)
        assert!(
            correct < total,
            "Wrong key should not decrypt all correctly"
        );
    }
}
