use lambdaworks_math::{traits::ByteConversion, unsigned_integer::element::UnsignedInteger};
use std::error::Error;
use std::fmt;

const NUM_LIMBS: usize = 4;

#[derive(Debug)]
pub enum RSAError {
    MessageTooLarge,
    InvalidBytes,
    InvalidCiphertext,
    NonInvertible, // when e is not invertible modulo φ(n)
}

impl fmt::Display for RSAError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RSAError::MessageTooLarge => write!(f, "Message must be less than n."),
            RSAError::InvalidBytes => write!(f, "Invalid bytes for conversion."),
            RSAError::InvalidCiphertext => write!(f, "Invalid ciphertext."),
            RSAError::NonInvertible => write!(f, "e is not invertible modulo φ(n)."),
        }
    }
}

impl Error for RSAError {}

/// Basic implementation of the RSA algorithm without panics.
pub struct RSA {
    pub e: UnsignedInteger<NUM_LIMBS>,
    pub d: UnsignedInteger<NUM_LIMBS>,
    pub n: UnsignedInteger<NUM_LIMBS>,
}

impl RSA {
    /// Generates an RSA instance from two primes `p` and `q`.
    /// Returns an error if the necessary conditions are not met.
    pub fn new(
        p: UnsignedInteger<NUM_LIMBS>,
        q: UnsignedInteger<NUM_LIMBS>,
    ) -> Result<Self, RSAError> {
        let n = p * q;

        // Compute φ(n) = (p - 1) * (q - 1)
        let phi_n = (p - UnsignedInteger::from_u64(1)) * (q - UnsignedInteger::from_u64(1));

        // Common public exponent.
        let e = UnsignedInteger::from_u64(65537);

        // Calculate d = e^(-1) mod φ(n)
        let d = Self::modinv(&e, &phi_n).ok_or(RSAError::NonInvertible)?;

        Ok(RSA { e, d, n })
    }

    /// Returns the public key (e, n)
    pub fn public_key(&self) -> (UnsignedInteger<NUM_LIMBS>, UnsignedInteger<NUM_LIMBS>) {
        (self.e, self.n)
    }

    /// Returns the private key d.
    pub fn secret_key(&self) -> UnsignedInteger<NUM_LIMBS> {
        self.d
    }

    /// Encrypts the message checking that message < n.
    pub fn encrypt(
        &self,
        message: &UnsignedInteger<NUM_LIMBS>,
    ) -> Result<UnsignedInteger<NUM_LIMBS>, RSAError> {
        if message >= &self.n {
            return Err(RSAError::MessageTooLarge);
        }

        Ok(modpow(message, &self.e, &self.n))
    }

    /// Decrypts the ciphertext checking that ciphertext < n.
    pub fn decrypt(
        &self,
        ciphertext: &UnsignedInteger<NUM_LIMBS>,
    ) -> Result<UnsignedInteger<NUM_LIMBS>, RSAError> {
        if ciphertext >= &self.n {
            return Err(RSAError::InvalidCiphertext);
        }

        Ok(modpow(ciphertext, &self.d, &self.n))
    }

    /// Encrypts a byte array without padding (for testing purposes only).
    #[cfg(feature = "alloc")]
    pub fn encrypt_bytes_simple(&self, msg: &[u8]) -> Result<Vec<u8>, RSAError> {
        // Create a fixed size vector (NUM_LIMBS * 8 bytes)
        let mut padded_msg = vec![0; NUM_LIMBS * 8];
        let msg_len = msg.len();
        if msg_len > padded_msg.len() {
            return Err(RSAError::MessageTooLarge);
        }
        padded_msg[NUM_LIMBS * 8 - msg_len..].copy_from_slice(msg);
        let m = UnsignedInteger::from_bytes_be(&padded_msg).map_err(|_| RSAError::InvalidBytes)?;
        let c = self.encrypt(&m)?;
        Ok(c.to_bytes_be())
    }

    /// Decrypts a byte array that was encrypted without padding.
    #[cfg(feature = "alloc")]
    pub fn decrypt_bytes_simple(&self, cipher: &[u8]) -> Result<Vec<u8>, RSAError> {
        let mut padded_cipher = vec![0; NUM_LIMBS * 8];
        let cipher_len = cipher.len();
        if cipher_len > padded_cipher.len() {
            return Err(RSAError::InvalidBytes);
        }
        padded_cipher[NUM_LIMBS * 8 - cipher_len..].copy_from_slice(cipher);
        let c =
            UnsignedInteger::from_bytes_be(&padded_cipher).map_err(|_| RSAError::InvalidBytes)?;
        let m = self.decrypt(&c)?;
        let decrypted = m.to_bytes_be();
        // Remove leading zeros to recover the original message
        let first_nonzero = decrypted
            .iter()
            .position(|&x| x != 0)
            .unwrap_or(decrypted.len());
        Ok(decrypted[first_nonzero..].to_vec())
    }

    /// Computes the modular inverse of `a` modulo `m` using the extended Euclidean algorithm.
    /// Returns None if a is not invertible.
    pub fn modinv(
        a: &UnsignedInteger<NUM_LIMBS>,
        m: &UnsignedInteger<NUM_LIMBS>,
    ) -> Option<UnsignedInteger<NUM_LIMBS>> {
        let mut t = UnsignedInteger::from_u64(0);
        let mut newt = UnsignedInteger::from_u64(1);
        let mut r = *m;
        let mut newr = *a;

        while newr != UnsignedInteger::from_u64(0) {
            let (quotient, _) = r.div_rem(&newr);
            let temp_t = t;
            t = newt;

            // Avoid underflow: perform subtraction in a modular way
            newt = if quotient * newt > temp_t {
                let diff = quotient * newt - temp_t;
                let (_, remainder) = diff.div_rem(m);
                *m - remainder
            } else {
                let diff = temp_t - quotient * newt;
                let (_, remainder) = diff.div_rem(m);
                remainder
            };

            let temp_r = r;
            r = newr;
            newr = temp_r - quotient * newr;
        }

        if r > UnsignedInteger::from_u64(1) {
            return None;
        }

        Some(t)
    }
}

/// Computes (base^exponent) mod modulus using the square-and-multiply algorithm.
fn modpow(
    base: &UnsignedInteger<NUM_LIMBS>,
    exponent: &UnsignedInteger<NUM_LIMBS>,
    modulus: &UnsignedInteger<NUM_LIMBS>,
) -> UnsignedInteger<NUM_LIMBS> {
    let mut result = UnsignedInteger::from_u64(1);
    let mut base = *base;
    let mut exponent = *exponent;

    while exponent != UnsignedInteger::from_u64(0) {
        if exponent.limbs[3] & 1 == 1 {
            result = (result * base).div_rem(modulus).1;
        }
        base = (base * base).div_rem(modulus).1;
        exponent >>= 1;
    }

    result
}
