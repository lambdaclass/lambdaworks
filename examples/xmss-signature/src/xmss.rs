//! XMSS (eXtended Merkle Signature Scheme) main API
//!
//! This module provides the high-level API for XMSS signatures:
//! - Key generation
//! - Signing messages
//! - Verifying signatures
//!
//! XMSS is a stateful hash-based signature scheme. The signer must
//! update the secret key after each signature to avoid reusing
//! one-time signatures.

use crate::address::{Address, AddressType};
use crate::hash::XmssHasher;
use crate::ltree::ltree;
use crate::params::{XmssParams, MAX_IDX, N};
use crate::wots::{wots_pk_from_sig, wots_sign, WotsSignature};
use crate::xmss_tree::{compute_root, AuthPath, XmssTree};

/// Error types for XMSS operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XmssError {
    /// All one-time signatures have been exhausted
    SignaturesExhausted,
    /// Invalid signature (verification failed)
    InvalidSignature,
    /// Invalid parameter
    InvalidParameter(String),
}

impl std::fmt::Display for XmssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XmssError::SignaturesExhausted => {
                write!(f, "All one-time signatures have been exhausted")
            }
            XmssError::InvalidSignature => write!(f, "Invalid signature"),
            XmssError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

impl std::error::Error for XmssError {}

/// XMSS Public Key
///
/// Contains the Merkle tree root and public seed.
#[derive(Clone, PartialEq, Eq)]
pub struct XmssPublicKey {
    /// The Merkle tree root
    pub root: [u8; N],
    /// Public seed for hash function
    pub public_seed: [u8; N],
}

impl XmssPublicKey {
    /// Create a new public key
    pub fn new(root: [u8; N], public_seed: [u8; N]) -> Self {
        Self { root, public_seed }
    }

    /// Serialize the public key to bytes
    pub fn to_bytes(&self) -> [u8; 2 * N] {
        let mut bytes = [0u8; 2 * N];
        bytes[0..N].copy_from_slice(&self.root);
        bytes[N..2 * N].copy_from_slice(&self.public_seed);
        bytes
    }

    /// Deserialize a public key from bytes
    pub fn from_bytes(bytes: &[u8; 2 * N]) -> Self {
        let mut root = [0u8; N];
        let mut public_seed = [0u8; N];
        root.copy_from_slice(&bytes[0..N]);
        public_seed.copy_from_slice(&bytes[N..2 * N]);
        Self { root, public_seed }
    }
}

/// XMSS Secret Key
///
/// Contains seeds for key derivation, the current index,
/// and cached tree data for efficient signing.
#[derive(Clone)]
pub struct XmssSecretKey {
    /// Secret seed for WOTS+ key derivation
    pub secret_seed: [u8; N],
    /// Secret seed for randomness generation in signing
    pub secret_prf: [u8; N],
    /// Public seed (also in public key, but needed for signing)
    pub public_seed: [u8; N],
    /// Current index (next available WOTS+ key pair)
    pub idx: u32,
    /// Cached tree for efficient auth path retrieval
    tree: Option<XmssTree>,
}

impl XmssSecretKey {
    /// Create a new secret key from seeds
    pub fn new(secret_seed: [u8; N], secret_prf: [u8; N], public_seed: [u8; N]) -> Self {
        Self {
            secret_seed,
            secret_prf,
            public_seed,
            idx: 0,
            tree: None,
        }
    }

    /// Get the current signature index
    pub fn index(&self) -> u32 {
        self.idx
    }

    /// Get remaining signatures available
    pub fn remaining_signatures(&self) -> u32 {
        if self.idx > MAX_IDX {
            0
        } else {
            MAX_IDX - self.idx + 1
        }
    }

    /// Build and cache the tree (for efficient auth path retrieval)
    fn ensure_tree<H: XmssHasher>(&mut self, hasher: &H) {
        if self.tree.is_none() {
            self.tree = Some(XmssTree::build(
                hasher,
                &self.secret_seed,
                &self.public_seed,
            ));
        }
    }

    /// Get the tree root
    pub fn root<H: XmssHasher>(&mut self, hasher: &H) -> [u8; N] {
        self.ensure_tree(hasher);
        *self.tree.as_ref().unwrap().root()
    }
}

/// XMSS Signature
///
/// Contains all data needed to verify a signature:
/// - The index of the WOTS+ key pair used
/// - Randomness used for message hashing
/// - The WOTS+ signature
/// - The authentication path
#[derive(Clone, Debug)]
pub struct XmssSignature {
    /// Index of the WOTS+ key pair used
    pub idx: u32,
    /// Randomness for message hash
    pub randomness: [u8; N],
    /// WOTS+ signature
    pub wots_sig: WotsSignature,
    /// Authentication path from leaf to root
    pub auth_path: AuthPath,
}

impl XmssSignature {
    /// Get the approximate size of the signature in bytes
    pub fn size(&self) -> usize {
        4 // idx
        + N // randomness
        + self.wots_sig.len() * N // WOTS+ signature
        + self.auth_path.path.len() * N // auth path
    }
}

/// XMSS signature scheme implementation
///
/// Generic over the hash function implementation.
pub struct Xmss<H: XmssHasher> {
    hasher: H,
    params: XmssParams,
}

impl<H: XmssHasher> Xmss<H> {
    /// Create a new XMSS instance with default parameters
    pub fn new(hasher: H) -> Self {
        Self {
            hasher,
            params: XmssParams::default(),
        }
    }

    /// Create a new XMSS instance with custom parameters
    pub fn with_params(hasher: H, params: XmssParams) -> Self {
        Self { hasher, params }
    }

    /// Get the parameters
    pub fn params(&self) -> &XmssParams {
        &self.params
    }

    /// Generate an XMSS key pair from a 96-byte seed
    ///
    /// The seed is split into:
    /// - Bytes 0-31: secret_seed (for WOTS+ key derivation)
    /// - Bytes 32-63: secret_prf (for randomness in signing)
    /// - Bytes 64-95: public_seed
    ///
    /// # Arguments
    /// * `seed` - 96 bytes of random data
    ///
    /// # Returns
    /// A tuple of (public_key, secret_key)
    pub fn keygen(&self, seed: &[u8; 96]) -> (XmssPublicKey, XmssSecretKey) {
        let mut secret_seed = [0u8; N];
        let mut secret_prf = [0u8; N];
        let mut public_seed = [0u8; N];

        secret_seed.copy_from_slice(&seed[0..N]);
        secret_prf.copy_from_slice(&seed[N..2 * N]);
        public_seed.copy_from_slice(&seed[2 * N..3 * N]);

        // Build the tree to get the root
        let tree = XmssTree::build(&self.hasher, &secret_seed, &public_seed);
        let root = *tree.root();

        let pk = XmssPublicKey::new(root, public_seed);
        let mut sk = XmssSecretKey::new(secret_seed, secret_prf, public_seed);
        sk.tree = Some(tree);

        (pk, sk)
    }

    /// Sign a message
    ///
    /// WARNING: This mutates the secret key by incrementing the index.
    /// Each WOTS+ key pair can only be used once. Reusing a key pair
    /// compromises security!
    ///
    /// # Arguments
    /// * `message` - The message to sign
    /// * `sk` - The secret key (will be mutated)
    ///
    /// # Returns
    /// The signature, or an error if signatures are exhausted
    pub fn sign(&self, message: &[u8], sk: &mut XmssSecretKey) -> Result<XmssSignature, XmssError> {
        // Check if we have signatures remaining
        if sk.idx > MAX_IDX {
            return Err(XmssError::SignaturesExhausted);
        }

        let idx = sk.idx;

        // Ensure tree is built
        sk.ensure_tree(&self.hasher);
        let tree = sk.tree.as_ref().unwrap();

        // Generate randomness for this signature
        let mut idx_bytes = [0u8; 32];
        idx_bytes[28..32].copy_from_slice(&idx.to_be_bytes());
        let randomness = self.hasher.prf(&sk.secret_prf, &idx_bytes);

        // Hash the message with randomness
        let msg_hash = self
            .hasher
            .h_msg(&randomness, tree.root(), idx as u64, message);

        // Create WOTS+ signature
        let mut address = Address::new();
        address.set_type(AddressType::Ots);
        address.set_ots_address(idx);

        let wots_sig = wots_sign(
            &self.hasher,
            &msg_hash,
            &sk.secret_seed,
            &sk.public_seed,
            &mut address,
        );

        // Get authentication path
        let auth_path = tree.auth_path(idx);

        // Increment index for next signature
        sk.idx += 1;

        Ok(XmssSignature {
            idx,
            randomness,
            wots_sig,
            auth_path,
        })
    }

    /// Verify a signature
    ///
    /// # Arguments
    /// * `message` - The message that was signed
    /// * `signature` - The signature to verify
    /// * `pk` - The public key
    ///
    /// # Returns
    /// true if the signature is valid, false otherwise
    pub fn verify(&self, message: &[u8], signature: &XmssSignature, pk: &XmssPublicKey) -> bool {
        // Check index is in valid range
        if signature.idx > MAX_IDX {
            return false;
        }

        // Recompute message hash
        let msg_hash = self.hasher.h_msg(
            &signature.randomness,
            &pk.root,
            signature.idx as u64,
            message,
        );

        // Compute WOTS+ public key from signature
        let mut address = Address::new();
        address.set_type(AddressType::Ots);
        address.set_ots_address(signature.idx);

        let wots_pk = wots_pk_from_sig(
            &self.hasher,
            &signature.wots_sig,
            &msg_hash,
            &pk.public_seed,
            &mut address,
        );

        // Compress WOTS+ public key with L-tree
        address.set_type(AddressType::LTree);
        address.set_ltree_address(signature.idx);
        let leaf = ltree(&self.hasher, &wots_pk, &pk.public_seed, &mut address);

        // Compute root from leaf and auth path
        let computed_root = compute_root(
            &self.hasher,
            &leaf,
            signature.idx,
            &signature.auth_path,
            &pk.public_seed,
        );

        // Verify root matches public key
        computed_root == pk.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Sha256Hasher;

    fn test_seed() -> [u8; 96] {
        let mut seed = [0u8; 96];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = i as u8;
        }
        seed
    }

    #[test]
    fn test_keygen() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk, sk) = xmss.keygen(&seed);

        assert_eq!(pk.root.len(), N);
        assert_eq!(pk.public_seed.len(), N);
        assert_eq!(sk.idx, 0);
    }

    #[test]
    fn test_keygen_deterministic() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk1, _) = xmss.keygen(&seed);
        let (pk2, _) = xmss.keygen(&seed);

        assert_eq!(pk1.root, pk2.root);
        assert_eq!(pk1.public_seed, pk2.public_seed);
    }

    #[test]
    fn test_sign_verify() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        let message = b"Hello, XMSS!";
        let signature = xmss.sign(message, &mut sk).unwrap();

        assert!(xmss.verify(message, &signature, &pk));
    }

    #[test]
    fn test_wrong_message_fails() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        let message = b"Hello, XMSS!";
        let wrong_message = b"Goodbye, XMSS!";

        let signature = xmss.sign(message, &mut sk).unwrap();

        assert!(!xmss.verify(wrong_message, &signature, &pk));
    }

    #[test]
    fn test_wrong_key_fails() {
        let xmss = Xmss::new(Sha256Hasher::new());

        let seed1 = test_seed();
        let mut seed2 = test_seed();
        seed2[0] = 0xFF;

        let (pk1, mut sk1) = xmss.keygen(&seed1);
        let (pk2, _) = xmss.keygen(&seed2);

        let message = b"Hello, XMSS!";
        let signature = xmss.sign(message, &mut sk1).unwrap();

        // Signature should verify with correct key
        assert!(xmss.verify(message, &signature, &pk1));

        // Signature should NOT verify with wrong key
        assert!(!xmss.verify(message, &signature, &pk2));
    }

    #[test]
    fn test_index_increments() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        assert_eq!(sk.idx, 0);

        let msg1 = b"Message 1";
        let sig1 = xmss.sign(msg1, &mut sk).unwrap();
        assert_eq!(sk.idx, 1);
        assert_eq!(sig1.idx, 0);

        let msg2 = b"Message 2";
        let sig2 = xmss.sign(msg2, &mut sk).unwrap();
        assert_eq!(sk.idx, 2);
        assert_eq!(sig2.idx, 1);

        // Both signatures should verify
        assert!(xmss.verify(msg1, &sig1, &pk));
        assert!(xmss.verify(msg2, &sig2, &pk));
    }

    #[test]
    fn test_multiple_signatures() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk, mut sk) = xmss.keygen(&seed);

        // Sign and verify multiple messages
        for i in 0..5 {
            let message = format!("Message number {}", i);
            let signature = xmss.sign(message.as_bytes(), &mut sk).unwrap();
            assert!(
                xmss.verify(message.as_bytes(), &signature, &pk),
                "Signature {} failed",
                i
            );
        }
    }

    #[test]
    fn test_signature_size() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (_, mut sk) = xmss.keygen(&seed);

        let message = b"Test message";
        let signature = xmss.sign(message, &mut sk).unwrap();

        // Expected size: 4 + 32 + 67*32 + 10*32 = 4 + 32 + 2144 + 320 = 2500 bytes
        let size = signature.size();
        assert!(
            size > 2000 && size < 3000,
            "Unexpected signature size: {}",
            size
        );
    }

    #[test]
    fn test_public_key_serialization() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (pk, _) = xmss.keygen(&seed);

        let bytes = pk.to_bytes();
        let pk_recovered = XmssPublicKey::from_bytes(&bytes);

        assert_eq!(pk.root, pk_recovered.root);
        assert_eq!(pk.public_seed, pk_recovered.public_seed);
    }

    #[test]
    fn test_remaining_signatures() {
        let xmss = Xmss::new(Sha256Hasher::new());
        let seed = test_seed();

        let (_, mut sk) = xmss.keygen(&seed);

        assert_eq!(sk.remaining_signatures(), 1024);

        let message = b"Test";
        xmss.sign(message, &mut sk).unwrap();

        assert_eq!(sk.remaining_signatures(), 1023);
    }
}
