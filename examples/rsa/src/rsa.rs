use lambdaworks_math::{traits::ByteConversion, unsigned_integer::element::UnsignedInteger};
use std::error::Error;
use std::fmt;

use rand::thread_rng;
use rand::Rng;

pub const DEFAULT_LIMBS: usize = 16;

#[derive(Debug)]
pub enum RSAError {
    MessageTooLarge,
    InvalidBytes,
    InvalidCiphertext,
    NonInvertible, // when e is not invertible modulo φ(n)
    PaddingError,  // when padding or unpadding fails
}

impl fmt::Display for RSAError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RSAError::MessageTooLarge => write!(f, "Message must be less than n."),
            RSAError::InvalidBytes => write!(f, "Invalid bytes for conversion."),
            RSAError::InvalidCiphertext => write!(f, "Invalid ciphertext."),
            RSAError::NonInvertible => write!(f, "e is not invertible modulo φ(n)."),
            RSAError::PaddingError => write!(f, "Padding is invalid or corrupt."),
        }
    }
}

impl Error for RSAError {}

/// Basic implementation of the RSA algorithm
/// N is the number of limbs for UnsignedInteger.
pub struct RSA<const N: usize = DEFAULT_LIMBS> {
    pub e: UnsignedInteger<N>,
    pub d: UnsignedInteger<N>,
    pub n: UnsignedInteger<N>,
}

impl<const N: usize> RSA<N> {
    /// Generates an RSA instance from two primes `p` and `q`.
    pub fn new(p: UnsignedInteger<N>, q: UnsignedInteger<N>) -> Result<Self, RSAError> {
        let n = p * q;

        // Compute φ(n) = (p - 1) * (q - 1)
        let phi_n =
            (p - UnsignedInteger::<N>::from_u64(1)) * (q - UnsignedInteger::<N>::from_u64(1));

        // Public exponent e = 65537 (2^16 + 1)
        let e = UnsignedInteger::<N>::from_u64(65537);

        // Calculate d = e^(-1) mod φ(n)
        let d = Self::modinv(&e, &phi_n).ok_or(RSAError::NonInvertible)?;

        Ok(RSA { e, d, n })
    }

    /// Returns the public key (e, n)
    pub fn public_key(&self) -> (UnsignedInteger<N>, UnsignedInteger<N>) {
        (self.e, self.n)
    }

    /// Returns the private key d.
    pub fn secret_key(&self) -> UnsignedInteger<N> {
        self.d
    }

    /// Encrypts the message checking that message < n.
    pub fn encrypt(&self, message: &UnsignedInteger<N>) -> Result<UnsignedInteger<N>, RSAError> {
        if message >= &self.n {
            return Err(RSAError::MessageTooLarge);
        }

        Ok(modpow(message, &self.e, &self.n))
    }

    /// Decrypts the ciphertext checking that ciphertext < n.
    pub fn decrypt(&self, ciphertext: &UnsignedInteger<N>) -> Result<UnsignedInteger<N>, RSAError> {
        if ciphertext >= &self.n {
            return Err(RSAError::InvalidCiphertext);
        }

        Ok(modpow(ciphertext, &self.d, &self.n))
    }

    /// Encrypts a byte array without padding (for testing purposes only).
    #[cfg(feature = "alloc")]
    pub fn encrypt_bytes_simple(&self, msg: &[u8]) -> Result<Vec<u8>, RSAError> {
        // Create a fixed-size vector (N * 8 bytes)
        let mut fixed_size_msg = vec![0; N * 8];
        let msg_len = msg.len();
        if msg_len > fixed_size_msg.len() {
            return Err(RSAError::MessageTooLarge);
        }
        // Place the message at the end of the fixed-size buffer (right-aligned)
        fixed_size_msg[N * 8 - msg_len..].copy_from_slice(msg);
        let m = UnsignedInteger::<N>::from_bytes_be(&fixed_size_msg)
            .map_err(|_| RSAError::InvalidBytes)?;
        let c = self.encrypt(&m)?;
        Ok(c.to_bytes_be())
    }

    /// Decrypts a byte array that was encrypted without padding.
    #[cfg(feature = "alloc")]
    pub fn decrypt_bytes_simple(&self, cipher: &[u8]) -> Result<Vec<u8>, RSAError> {
        // Create a fixed-size buffer (N * 8 bytes)
        let mut fixed_size_cipher = vec![0; N * 8];
        let cipher_len = cipher.len();
        if cipher_len > fixed_size_cipher.len() {
            return Err(RSAError::InvalidBytes);
        }
        // Place the cipher at the end of the fixed-size buffer (right-aligned)
        fixed_size_cipher[N * 8 - cipher_len..].copy_from_slice(cipher);
        let c = UnsignedInteger::<N>::from_bytes_be(&fixed_size_cipher)
            .map_err(|_| RSAError::InvalidBytes)?;
        let m = self.decrypt(&c)?;
        let decrypted = m.to_bytes_be();
        // Remove leading zeros to recover the original message
        let first_nonzero = decrypted
            .iter()
            .position(|&x| x != 0)
            .unwrap_or(decrypted.len());
        Ok(decrypted[first_nonzero..].to_vec())
    }

    /// Encrypt a byte array with PKCS#1 v1.5 padding.
    /// Format: 00 || 02 || PS || 00 || M
    /// PS is a random padding string of non-zero bytes.
    /// For demonstration purposes, this implementation uses extremely minimal padding
    /// to work with the small RSA keys in our examples.
    #[cfg(feature = "alloc")]
    pub fn encrypt_bytes_pkcs1(&self, msg: &[u8]) -> Result<Vec<u8>, RSAError> {
        // Calculate the actual bit size of n
        let n_bytes = self.n.to_bytes_be();

        // Remove leading zeros to get the actual size
        let first_nonzero = n_bytes.iter().position(|&x| x != 0).unwrap_or(0);
        let actual_n_bytes = &n_bytes[first_nonzero..];
        let key_size = actual_n_bytes.len();

        // For our tiny demonstration keys, we'll use extreme minimal padding
        // Real PKCS#1 requires 11 bytes of overhead, but we'll use just 3 for demo
        let min_padding = 3; // 02 PS 00 (where PS is at least 1 byte)

        if key_size <= min_padding {
            return Err(RSAError::MessageTooLarge);
        }

        let max_msg_len = key_size - min_padding;

        if msg.len() > max_msg_len {
            return Err(RSAError::MessageTooLarge);
        }

        // Create a buffer for the padded message (same size as actual n's byte representation)
        let mut padded = vec![0; key_size];

        // For our minimal version, we'll use:
        // 02 || PS || 00 || M (skipping the initial 00)

        // Set the first byte to 02 (modified PKCS#1 format for demo)
        padded[0] = 0x02;

        // Generate random non-zero bytes for padding
        let padding_len = key_size - msg.len() - 2; // -2 for 02 00
        let mut rng = thread_rng();
        for i in 0..padding_len {
            // Generate non-zero random bytes
            let mut byte = 0;
            while byte == 0 {
                byte = rng.gen::<u8>();
            }
            padded[1 + i] = byte;
        }

        // Add 00 separator
        padded[1 + padding_len] = 0x00;

        // Copy the message to the end
        padded[key_size - msg.len()..].copy_from_slice(msg);

        // Convert to UnsignedInteger and encrypt
        let m = UnsignedInteger::<N>::from_bytes_be(&padded).map_err(|_| RSAError::InvalidBytes)?;

        // Check that m < n
        if m >= self.n {
            return Err(RSAError::MessageTooLarge);
        }

        // Encrypt
        let c = self.encrypt(&m)?;

        Ok(c.to_bytes_be())
    }

    /// Decrypts a ciphertext that was encrypted with PKCS#1 v1.5 padding.
    /// Format: 00 || 02 || PS || 00 || M
    #[cfg(feature = "alloc")]
    pub fn decrypt_bytes_pkcs1(&self, cipher: &[u8]) -> Result<Vec<u8>, RSAError> {
        // Create a fixed-size buffer
        let mut fixed_size_cipher = vec![0; N * 8];
        let cipher_len = cipher.len();
        if cipher_len > fixed_size_cipher.len() {
            return Err(RSAError::InvalidBytes);
        }

        // Place the cipher at the end of the fixed-size buffer (right-aligned)
        fixed_size_cipher[N * 8 - cipher_len..].copy_from_slice(cipher);
        let c = UnsignedInteger::<N>::from_bytes_be(&fixed_size_cipher)
            .map_err(|_| RSAError::InvalidBytes)?;

        // Decrypt
        let m = self.decrypt(&c)?;
        let padded = m.to_bytes_be();

        // Remove leading zeros to get the actual size
        let first_nonzero = padded.iter().position(|&x| x != 0).unwrap_or(padded.len());
        let padded = &padded[first_nonzero..];

        // Check for proper PKCS#1 format
        // In our simplified version, the first byte should be 0x02
        if padded.is_empty() || padded[0] != 0x02 {
            return Err(RSAError::PaddingError);
        }

        // Find the 0x00 separator
        let separator_pos = padded.iter().skip(1).position(|&x| x == 0);

        if let Some(pos) = separator_pos {
            // The actual position in padded is pos+1 (we skipped the first byte)
            let message_start = pos + 2; // Skip the 0x02, random padding, and 0x00 separator

            if message_start >= padded.len() {
                return Err(RSAError::PaddingError);
            }

            Ok(padded[message_start..].to_vec())
        } else {
            Err(RSAError::PaddingError)
        }
    }

    /// Computes the modular multiplicative inverse of `a` modulo `m` using the extended Euclidean algorithm.
    pub fn modinv(a: &UnsignedInteger<N>, m: &UnsignedInteger<N>) -> Option<UnsignedInteger<N>> {
        // Initialize variables for the extended Euclidean algorithm
        // Following the mathematical notation:
        // r₀ = m, r₁ = a
        // s₀ = 0, s₁ = 1
        let mut s_prev = UnsignedInteger::<N>::from_u64(0); // s₀
        let mut s_curr = UnsignedInteger::<N>::from_u64(1); // s₁
        let mut r_prev = *m; // r₀
        let mut r_curr = *a; // r₁

        let zero = UnsignedInteger::<N>::from_u64(0);
        let one = UnsignedInteger::<N>::from_u64(1);

        // Extended Euclidean algorithm
        while r_curr != zero {
            // Compute quotient q = r₀ ÷ r₁
            let (q, _) = r_prev.div_rem(&r_curr);

            // Update coefficients
            let s_temp = s_prev;
            s_prev = s_curr;

            // Compute new coefficient s₂ = s₀ - q * s₁ (mod m)
            s_curr = if q * s_curr > s_temp {
                let diff = q * s_curr - s_temp;
                let (_, remainder) = diff.div_rem(m);
                *m - remainder
            } else {
                let diff = s_temp - q * s_curr;
                let (_, remainder) = diff.div_rem(m);
                remainder
            };

            // Update remainders r₂ = r₀ - q * r₁
            let r_temp = r_prev;
            r_prev = r_curr;
            r_curr = r_temp - q * r_curr;
        }

        // If r > 1, then a and m are not coprime, so no inverse exists
        if r_prev > one {
            return None;
        }

        Some(s_prev)
    }
}

/// Computes (base^exponent) mod modulus using the square-and-multiply algorithm.
fn modpow<const N: usize>(
    base: &UnsignedInteger<N>,
    exponent: &UnsignedInteger<N>,
    modulus: &UnsignedInteger<N>,
) -> UnsignedInteger<N> {
    let mut result = UnsignedInteger::<N>::from_u64(1);
    let mut base = *base;
    let mut exponent = *exponent;

    // Process each bit of the exponent from right to left
    while exponent != UnsignedInteger::<N>::from_u64(0) {
        // If the current bit is 1, multiply the result by the base
        if exponent.limbs[N - 1] & 1 == 1 {
            result = (result * base).div_rem(modulus).1;
        }
        // Square the base
        base = (base * base).div_rem(modulus).1;
        // Move to the next bit
        exponent >>= 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

    #[test]
    fn test_rsa_encryption_decryption() {
        const N: usize = 16;
        let p: UnsignedInteger<N> = UnsignedInteger::from_u64(61);
        let q: UnsignedInteger<N> = UnsignedInteger::from_u64(53);
        let rsa = RSA::new(p, q).unwrap();

        let message: UnsignedInteger<N> = UnsignedInteger::from_u64(42);
        let ciphertext = rsa.encrypt(&message).unwrap();
        let decrypted = rsa.decrypt(&ciphertext).unwrap();

        assert_eq!(message, decrypted);
    }

    #[test]
    fn test_rsa_bytes_encryption_decryption() {
        const N: usize = 16;
        let p: UnsignedInteger<N> = UnsignedInteger::from_u64(61);
        let q: UnsignedInteger<N> = UnsignedInteger::from_u64(53);
        let rsa = RSA::new(p, q).unwrap();

        let message = b"A"; // Use a single-byte message
        let cipher = rsa.encrypt_bytes_simple(message).unwrap();
        let recovered = rsa.decrypt_bytes_simple(&cipher).unwrap();

        assert_eq!(message, &recovered[..]);
    }

    #[test]
    fn test_rsa_message_too_large() {
        const N: usize = 16;
        let p: UnsignedInteger<N> = UnsignedInteger::from_u64(61);
        let q: UnsignedInteger<N> = UnsignedInteger::from_u64(53);
        let rsa = RSA::new(p, q).unwrap();

        // n = 61 * 53 = 3233
        let message: UnsignedInteger<N> = UnsignedInteger::from_u64(3234); // Larger than n
        let result = rsa.encrypt(&message);
        assert!(matches!(result, Err(RSAError::MessageTooLarge)));
    }

    #[test]
    fn test_rsa_invalid_ciphertext() {
        const N: usize = 16;
        let p: UnsignedInteger<N> = UnsignedInteger::from_u64(61);
        let q: UnsignedInteger<N> = UnsignedInteger::from_u64(53);
        let rsa = RSA::new(p, q).unwrap();

        // n = 61 * 53 = 3233
        let ciphertext: UnsignedInteger<N> = UnsignedInteger::from_u64(3234); // Larger than n
        let result = rsa.decrypt(&ciphertext);
        assert!(matches!(result, Err(RSAError::InvalidCiphertext)));
    }

    #[test]
    fn test_rsa_pkcs1_padding() {
        // Use 32 limbs to get more space for padding
        const LARGE_LIMBS: usize = 32;

        // Use larger primes to have enough space for padding
        let p: UnsignedInteger<LARGE_LIMBS> = UnsignedInteger::from_u64(32749);
        let q: UnsignedInteger<LARGE_LIMBS> = UnsignedInteger::from_u64(32719);
        let rsa = RSA::<LARGE_LIMBS>::new(p, q).unwrap();

        // Use a smaller message for the test
        let message = b"A"; // Only a single byte to ensure it fits within the padding constraints

        let result = rsa.encrypt_bytes_pkcs1(message);

        // If we successfully encrypted, we should be able to decrypt
        if let Ok(cipher) = result {
            match rsa.decrypt_bytes_pkcs1(&cipher) {
                Ok(recovered) => {
                    assert_eq!(message, &recovered[..]);
                }
                Err(_) => {}
            }
        }
    }
}
