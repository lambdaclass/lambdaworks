//! WOTS+ (Winternitz One-Time Signature Plus) implementation
//!
//! WOTS+ is a hash-based one-time signature scheme where:
//! - Secret key: LEN random n-byte values
//! - Public key: Each secret key element chained (w-1) times through F
//! - Signature: Each element chained a number of times based on message digit
//! - Verification: Complete the chains and compare to public key

use crate::address::{Address, AddressType};
use crate::hash::XmssHasher;
use crate::params::{LEN, N, W};
use crate::utils::msg_to_wots_input;

/// WOTS+ signature containing LEN n-byte chain values
#[derive(Clone, Debug)]
pub struct WotsSignature {
    /// The signature values (one per chain)
    pub sig: Vec<[u8; N]>,
}

impl WotsSignature {
    /// Create a new WOTS+ signature from raw values
    pub fn new(sig: Vec<[u8; N]>) -> Self {
        assert_eq!(sig.len(), LEN, "WOTS+ signature must have {} elements", LEN);
        Self { sig }
    }

    /// Get the number of elements in the signature
    pub fn len(&self) -> usize {
        self.sig.len()
    }

    /// Check if the signature is empty
    pub fn is_empty(&self) -> bool {
        self.sig.is_empty()
    }
}

/// WOTS+ public key containing LEN n-byte values
#[derive(Clone, PartialEq, Eq)]
pub struct WotsPublicKey {
    /// The public key values (end of each chain)
    pub pk: Vec<[u8; N]>,
}

impl WotsPublicKey {
    /// Create a new WOTS+ public key
    pub fn new(pk: Vec<[u8; N]>) -> Self {
        assert_eq!(pk.len(), LEN, "WOTS+ public key must have {} elements", LEN);
        Self { pk }
    }
}

/// Compute the chain function: apply F iteratively
///
/// chain(X, i, s, SEED, ADRS) computes F^s(X) starting from position i
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `input` - Starting value X
/// * `start` - Starting index i in the chain
/// * `steps` - Number of iterations s
/// * `public_seed` - Public seed for F
/// * `address` - Address for domain separation (will be modified)
///
/// # Returns
/// The result after applying F `steps` times
pub fn chain<H: XmssHasher>(
    hasher: &H,
    input: &[u8; N],
    start: u32,
    steps: u32,
    public_seed: &[u8; N],
    address: &mut Address,
) -> [u8; N] {
    // If no steps, return input unchanged
    if steps == 0 {
        return *input;
    }

    let mut result = *input;

    for i in start..(start + steps) {
        address.set_hash_address(i);
        result = hasher.f(&result, public_seed, address);
    }

    result
}

/// Generate WOTS+ secret key element from seed
///
/// Uses PRF_keygen to derive the i-th secret key element
fn wots_sk_element<H: XmssHasher>(
    hasher: &H,
    secret_seed: &[u8; N],
    address: &mut Address,
    chain_idx: u32,
) -> [u8; N] {
    address.set_chain_address(chain_idx);
    address.set_hash_address(0);
    address.set_key_and_mask(0);
    hasher.prf_keygen(secret_seed, address)
}

/// Generate WOTS+ public key from secret seed
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `secret_seed` - Secret seed S
/// * `public_seed` - Public seed SEED
/// * `address` - OTS address (specifies which WOTS+ instance)
///
/// # Returns
/// The WOTS+ public key
pub fn wots_pkgen<H: XmssHasher>(
    hasher: &H,
    secret_seed: &[u8; N],
    public_seed: &[u8; N],
    address: &mut Address,
) -> WotsPublicKey {
    address.set_type(AddressType::Ots);

    let mut pk = Vec::with_capacity(LEN);

    for i in 0..LEN {
        // Generate secret key element
        let sk_i = wots_sk_element(hasher, secret_seed, address, i as u32);

        // Chain it (w-1) times to get public key element
        address.set_chain_address(i as u32);
        let pk_i = chain(hasher, &sk_i, 0, (W - 1) as u32, public_seed, address);
        pk.push(pk_i);
    }

    WotsPublicKey::new(pk)
}

/// Sign a message hash using WOTS+
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `msg_hash` - The message hash (32 bytes)
/// * `secret_seed` - Secret seed S
/// * `public_seed` - Public seed SEED
/// * `address` - OTS address (specifies which WOTS+ instance)
///
/// # Returns
/// The WOTS+ signature
pub fn wots_sign<H: XmssHasher>(
    hasher: &H,
    msg_hash: &[u8; N],
    secret_seed: &[u8; N],
    public_seed: &[u8; N],
    address: &mut Address,
) -> WotsSignature {
    address.set_type(AddressType::Ots);

    // Convert message to WOTS+ input (message digits + checksum)
    let msg_base_w = msg_to_wots_input(msg_hash);

    let mut sig = Vec::with_capacity(LEN);

    for (i, &msg_val) in msg_base_w.iter().enumerate().take(LEN) {
        // Generate secret key element
        let sk_i = wots_sk_element(hasher, secret_seed, address, i as u32);

        // Chain it msg[i] times
        address.set_chain_address(i as u32);
        let sig_i = chain(hasher, &sk_i, 0, msg_val, public_seed, address);
        sig.push(sig_i);
    }

    WotsSignature::new(sig)
}

/// Compute WOTS+ public key from signature
///
/// This is the verification helper that computes what the public key
/// should be given a signature and message.
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `signature` - The WOTS+ signature
/// * `msg_hash` - The message hash (32 bytes)
/// * `public_seed` - Public seed SEED
/// * `address` - OTS address (specifies which WOTS+ instance)
///
/// # Returns
/// The computed public key (should match the actual public key if valid)
pub fn wots_pk_from_sig<H: XmssHasher>(
    hasher: &H,
    signature: &WotsSignature,
    msg_hash: &[u8; N],
    public_seed: &[u8; N],
    address: &mut Address,
) -> WotsPublicKey {
    address.set_type(AddressType::Ots);

    // Convert message to WOTS+ input
    let msg_base_w = msg_to_wots_input(msg_hash);

    let mut pk = Vec::with_capacity(LEN);

    for (i, &msg_val) in msg_base_w.iter().enumerate().take(LEN) {
        // Complete the chain from signature value to public key
        // We need (w - 1 - msg[i]) more steps
        address.set_chain_address(i as u32);
        let steps = (W as u32) - 1 - msg_val;
        let pk_i = chain(
            hasher,
            &signature.sig[i],
            msg_val,
            steps,
            public_seed,
            address,
        );
        pk.push(pk_i);
    }

    WotsPublicKey::new(pk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Sha256Hasher;

    #[test]
    fn test_chain_zero_steps() {
        let hasher = Sha256Hasher::new();
        let input = [0xABu8; N];
        let seed = [0x42u8; N];
        let mut address = Address::new();
        address.set_type(AddressType::Ots);

        let result = chain(&hasher, &input, 0, 0, &seed, &mut address);
        assert_eq!(result, input);
    }

    #[test]
    fn test_chain_one_step() {
        let hasher = Sha256Hasher::new();
        let input = [0xABu8; N];
        let seed = [0x42u8; N];
        let mut address = Address::new();
        address.set_type(AddressType::Ots);

        let result = chain(&hasher, &input, 0, 1, &seed, &mut address);
        assert_ne!(result, input);
    }

    #[test]
    fn test_chain_deterministic() {
        let hasher = Sha256Hasher::new();
        let input = [0xABu8; N];
        let seed = [0x42u8; N];

        let mut addr1 = Address::new();
        addr1.set_type(AddressType::Ots);
        let result1 = chain(&hasher, &input, 0, 5, &seed, &mut addr1);

        let mut addr2 = Address::new();
        addr2.set_type(AddressType::Ots);
        let result2 = chain(&hasher, &input, 0, 5, &seed, &mut addr2);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_wots_sign_verify() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];
        let msg_hash = [0x03u8; N];

        let mut address = Address::new();
        address.set_ots_address(0);

        // Generate public key
        let pk = wots_pkgen(&hasher, &secret_seed, &public_seed, &mut address);

        // Sign the message
        let sig = wots_sign(&hasher, &msg_hash, &secret_seed, &public_seed, &mut address);

        // Recover public key from signature
        let recovered_pk = wots_pk_from_sig(&hasher, &sig, &msg_hash, &public_seed, &mut address);

        // Should match
        assert_eq!(pk.pk, recovered_pk.pk);
    }

    #[test]
    fn test_wots_wrong_message_fails() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];
        let msg_hash = [0x03u8; N];
        let wrong_msg = [0x04u8; N];

        let mut address = Address::new();
        address.set_ots_address(0);

        // Generate public key
        let pk = wots_pkgen(&hasher, &secret_seed, &public_seed, &mut address);

        // Sign the message
        let sig = wots_sign(&hasher, &msg_hash, &secret_seed, &public_seed, &mut address);

        // Try to verify with wrong message
        let recovered_pk = wots_pk_from_sig(&hasher, &sig, &wrong_msg, &public_seed, &mut address);

        // Should NOT match
        assert_ne!(pk.pk, recovered_pk.pk);
    }

    #[test]
    fn test_wots_signature_length() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];
        let msg_hash = [0x03u8; N];

        let mut address = Address::new();
        address.set_ots_address(0);

        let sig = wots_sign(&hasher, &msg_hash, &secret_seed, &public_seed, &mut address);
        assert_eq!(sig.len(), LEN);
    }

    #[test]
    fn test_different_ots_addresses_different_keys() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let mut address1 = Address::new();
        address1.set_ots_address(0);
        let pk1 = wots_pkgen(&hasher, &secret_seed, &public_seed, &mut address1);

        let mut address2 = Address::new();
        address2.set_ots_address(1);
        let pk2 = wots_pkgen(&hasher, &secret_seed, &public_seed, &mut address2);

        // Different OTS addresses should give different public keys
        assert_ne!(pk1.pk, pk2.pk);
    }
}
