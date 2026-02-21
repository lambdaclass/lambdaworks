//! Fp3-specific round result types for the GPU STARK prover.
//!
//! These types support the mixed F/E field architecture where:
//! - Main trace operations use the base field (Goldilocks64Field)
//! - Auxiliary trace, composition poly, and FRI use the extension field (Fp3)

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::{
    Degree3GoldilocksExtensionField, Goldilocks64Field,
};
use lambdaworks_math::polynomial::Polynomial;

use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};
use stark_platinum_prover::fri::fri_decommit::FriDecommitment;
use stark_platinum_prover::proof::stark::DeepPolynomialOpening;
use stark_platinum_prover::table::Table;

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::Buffer;

type F = Goldilocks64Field;
type Fp3 = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;
type Fp3E = FieldElement<Fp3>;

/// Result of GPU Phase 1 (RAP round) for Fp3 extension field proofs.
///
/// Main trace data is in the base field (F), auxiliary trace data is in Fp3.
pub struct GpuRound1ResultFp3 {
    /// Polynomial coefficients from interpolating main trace columns (base field).
    pub main_trace_polys: Vec<Polynomial<FE>>,
    /// Polynomial coefficients from interpolating auxiliary trace columns (Fp3).
    pub aux_trace_polys: Vec<Polynomial<Fp3E>>,
    /// LDE evaluations of main trace (column-major, base field).
    pub main_lde_evaluations: Vec<Vec<FE>>,
    /// LDE evaluations of auxiliary trace (column-major, Fp3).
    pub aux_lde_evaluations: Vec<Vec<Fp3E>>,
    /// GPU buffers for main trace LDE (base field) - kept on GPU for Phase 2/4.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub main_lde_buffers: Vec<Buffer>,
    /// GPU buffers for auxiliary trace LDE (Fp3) - kept on GPU for Phase 2/4.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub aux_lde_buffers: Vec<Buffer>,
    /// Size of LDE domain (number of elements per column).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub lde_domain_size: usize,
    /// Merkle tree for main trace LDE (base field).
    pub main_merkle_tree: BatchedMerkleTree<F>,
    /// Merkle root for main trace.
    pub main_merkle_root: Commitment,
    /// Merkle tree for auxiliary trace LDE (Fp3).
    pub aux_merkle_tree: Option<BatchedMerkleTree<Fp3>>,
    /// Merkle root for auxiliary trace.
    pub aux_merkle_root: Option<Commitment>,
    /// RAP challenges sampled from the transcript (Fp3).
    pub rap_challenges: Vec<Fp3E>,
}

/// Result of GPU Phase 2 (composition polynomial round) for Fp3 extension field proofs.
///
/// All composition polynomial data is in Fp3.
pub struct GpuRound2ResultFp3 {
    /// The composition polynomial broken into parts (Fp3).
    pub composition_poly_parts: Vec<Polynomial<Fp3E>>,
    /// LDE evaluations of each composition poly part (column-major, Fp3).
    pub lde_composition_poly_evaluations: Vec<Vec<Fp3E>>,
    /// Merkle tree for the composition polynomial commitment (Fp3).
    pub composition_poly_merkle_tree: BatchedMerkleTree<Fp3>,
    /// Commitment root.
    pub composition_poly_root: Commitment,
}

/// Result of GPU Phase 3 (OOD evaluation round) for Fp3 extension field proofs.
pub struct GpuRound3ResultFp3 {
    /// OOD evaluations of trace polynomials (in Fp3).
    pub trace_ood_evaluations: Table<Fp3>,
    /// OOD evaluations of composition polynomial parts at z^N (Fp3).
    pub composition_poly_parts_ood_evaluation: Vec<Fp3E>,
    /// The out-of-domain challenge point z (Fp3).
    pub z: Fp3E,
}

/// Result of GPU Phase 4 (FRI round) for Fp3 extension field proofs.
pub struct GpuRound4ResultFp3 {
    /// The final constant value from FRI folding (Fp3).
    pub fri_last_value: Fp3E,
    /// Merkle roots of each FRI inner layer.
    pub fri_layers_merkle_roots: Vec<Commitment>,
    /// Merkle opening proofs for trace and composition polynomial evaluations.
    pub deep_poly_openings: Vec<DeepPolynomialOpening<F, Fp3>>,
    /// FRI query decommitments (Fp3).
    pub query_list: Vec<FriDecommitment<Fp3>>,
    /// Grinding nonce (None if grinding_factor == 0).
    pub nonce: Option<u64>,
}
