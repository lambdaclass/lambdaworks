//! FRI data structures: configuration, layers, proofs, errors.

use lambdaworks_crypto::merkle_tree::proof::Proof as MerkleProof;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// FRI protocol configuration.
#[derive(Debug, Clone)]
pub struct FriConfig {
    /// log2 of the blowup factor. 1 means 2x blowup, 2 means 4x, etc.
    pub log_blowup: usize,
    /// Number of query repetitions (security parameter).
    pub num_queries: usize,
}

impl FriConfig {
    /// Maximum supported log2(blowup) to prevent unreasonable memory use.
    const MAX_LOG_BLOWUP: usize = 8; // 256x blowup

    /// Maximum supported log2(degree_bound) for LDE allocation.
    pub const MAX_LOG_DEGREE: usize = 28; // 2^28 = 256M field elements

    /// Creates a validated FRI configuration.
    pub fn new(log_blowup: usize, num_queries: usize) -> Result<Self, FriError> {
        if log_blowup == 0 || log_blowup > Self::MAX_LOG_BLOWUP {
            return Err(FriError::InvalidConfig(format!(
                "log_blowup must be in [1, {}], got {log_blowup}",
                Self::MAX_LOG_BLOWUP
            )));
        }
        if num_queries == 0 {
            return Err(FriError::InvalidConfig("num_queries must be >= 1".into()));
        }
        Ok(Self {
            log_blowup,
            num_queries,
        })
    }

    pub fn blowup_factor(&self) -> usize {
        1 << self.log_blowup
    }
}

impl Default for FriConfig {
    fn default() -> Self {
        Self {
            log_blowup: 1,   // 2x blowup
            num_queries: 30, // ~30 bits of security per query
        }
    }
}

/// Proof data for a single FRI query at one layer.
#[derive(Debug, Clone)]
pub struct FriQueryRound<F: IsField> {
    /// Evaluation at the query index.
    pub eval: FieldElement<F>,
    /// Evaluation at the symmetric index (index XOR half_domain_size).
    pub eval_sym: FieldElement<F>,
    /// Merkle authentication path for the query index.
    pub auth_path: MerkleProof<[u8; 32]>,
    /// Merkle authentication path for the symmetric index.
    pub auth_path_sym: MerkleProof<[u8; 32]>,
}

/// Complete FRI proof.
#[derive(Debug, Clone)]
pub struct FriProof<F: IsField> {
    /// Merkle roots for each layer (sent to verifier).
    pub layer_merkle_roots: Vec<[u8; 32]>,
    /// Query decommitments: outer = per query, inner = per layer.
    pub query_rounds: Vec<Vec<FriQueryRound<F>>>,
    /// The final constant polynomial value.
    pub final_value: FieldElement<F>,
}

/// Decommitment for an original (pre-FRI) polynomial at a query point.
///
/// Contains evaluations and Merkle authentication paths at both the query
/// index and its symmetric partner in the LDE domain.
#[derive(Debug, Clone)]
pub struct OriginalPolyDecommitment<F: IsField> {
    /// Evaluation at the query index.
    pub eval: FieldElement<F>,
    /// Evaluation at the symmetric partner (index + half_domain).
    pub eval_sym: FieldElement<F>,
    /// Merkle authentication path for the query index.
    pub auth_path: MerkleProof<[u8; 32]>,
    /// Merkle authentication path for the symmetric index.
    pub auth_path_sym: MerkleProof<[u8; 32]>,
}

/// Prover-side data for a committed FRI layer.
pub struct FriLayerData<F: IsField> {
    /// Merkle root.
    pub merkle_root: [u8; 32],
    /// LDE evaluations at this layer.
    pub evaluations: Vec<FieldElement<F>>,
    /// Domain size for this layer's LDE (= poly_degree_bound * blowup).
    pub domain_size: usize,
}

/// FRI error types.
#[derive(Debug, Clone)]
pub enum FriError {
    /// Polynomial degree too large for the domain.
    DegreeTooLarge { degree: usize, max: usize },
    /// FFT operation failed.
    FftError(String),
    /// Merkle tree construction failed.
    MerkleError(String),
    /// Verification: fold consistency check failed at a query.
    FoldConsistencyFailed { query: usize, layer: usize },
    /// Verification: Merkle proof failed.
    MerkleProofFailed { query: usize, layer: usize },
    /// Verification: final value mismatch.
    FinalValueMismatch,
    /// Invalid configuration.
    InvalidConfig(String),
}

impl core::fmt::Display for FriError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DegreeTooLarge { degree, max } => {
                write!(f, "polynomial degree {degree} exceeds max {max}")
            }
            Self::FftError(s) => write!(f, "FFT error: {s}"),
            Self::MerkleError(s) => write!(f, "Merkle error: {s}"),
            Self::FoldConsistencyFailed { query, layer } => {
                write!(f, "fold consistency failed at query {query}, layer {layer}")
            }
            Self::MerkleProofFailed { query, layer } => {
                write!(f, "Merkle proof failed at query {query}, layer {layer}")
            }
            Self::FinalValueMismatch => write!(f, "final constant value mismatch"),
            Self::InvalidConfig(s) => write!(f, "invalid FRI config: {s}"),
        }
    }
}
