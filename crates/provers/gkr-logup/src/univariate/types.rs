use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

use super::super::verifier::Proof;
use super::domain::CyclicDomainError;
use super::pcs::PcsError;

/// Proof for the univariate IOP (Section 5 of ePrint 2023/1284).
///
/// Phase 1: transparent commitments (raw values in Fiat-Shamir transcript).
/// Bridges multilinear GKR evaluation claims to univariate polynomial commitments.
/// The verifier checks that committed univariate polynomials are consistent with the
/// GKR output via Lagrange column inner products.
#[derive(Debug, Clone)]
pub struct UnivariateIopProof<F: IsField> {
    /// Committed univariate polynomial values (Phase 1: raw values as Fiat-Shamir commitment).
    pub committed_columns: Vec<Vec<FieldElement<F>>>,
    /// The standard GKR proof over the multilinear representation.
    pub gkr_proof: Proof<F>,
    /// Lagrange column values `c_i = eq(iota(i), t)` for the evaluation point `t`.
    pub lagrange_column: Vec<FieldElement<F>>,
}

/// Proof for the univariate IOP Phase 2: PCS-based commitments + univariate sumcheck.
///
/// Instead of raw polynomial values, uses cryptographic commitments (e.g. FRI Merkle roots).
/// The inner product check is replaced by a univariate sumcheck verified at a random point.
#[derive(Debug, Clone)]
pub struct UnivariateIopProofV2<
    F: IsField,
    C: Clone + core::fmt::Debug,
    B: Clone + core::fmt::Debug,
> {
    /// PCS commitments for each input column.
    pub column_commitments: Vec<C>,
    /// The standard GKR proof over the multilinear representation.
    pub gkr_proof: Proof<F>,
    /// PCS commitment for the sumcheck quotient polynomial q(X).
    pub q_commitment: C,
    /// PCS commitment for the sumcheck remainder polynomial r'(X).
    pub r_prime_commitment: C,
    /// Batch opening proof for all polynomials at z.
    pub batch_proof: B,
    /// Opened values at z: [col_0(z), col_1(z), ..., q(z), r'(z)].
    pub opened_values: Vec<FieldElement<F>>,
}

/// Error type for the univariate IOP.
#[derive(Debug)]
pub enum UnivariateIopError {
    /// Lagrange column constraint verification failed.
    LagrangeConstraintFailed(super::lagrange_column::LagrangeColumnError),
    /// Inner product of committed values and Lagrange column doesn't match the GKR claim.
    InnerProductMismatch,
    /// Univariate sumcheck equation check failed.
    SumcheckFailed,
    /// GKR prover or verifier returned an error.
    GkrError(String),
    /// Cyclic domain creation failed.
    DomainError(CyclicDomainError),
    /// PCS operation failed.
    PcsError(PcsError),
}

impl core::fmt::Display for UnivariateIopError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LagrangeConstraintFailed(e) => {
                write!(f, "Lagrange column constraint failed: {e}")
            }
            Self::InnerProductMismatch => {
                write!(
                    f,
                    "inner product mismatch: committed values inconsistent with GKR claims"
                )
            }
            Self::SumcheckFailed => {
                write!(f, "univariate sumcheck equation check failed")
            }
            Self::GkrError(msg) => write!(f, "GKR error: {msg}"),
            Self::DomainError(e) => write!(f, "domain error: {e}"),
            Self::PcsError(e) => write!(f, "PCS error: {e}"),
        }
    }
}

impl From<PcsError> for UnivariateIopError {
    fn from(e: PcsError) -> Self {
        Self::PcsError(e)
    }
}

impl From<CyclicDomainError> for UnivariateIopError {
    fn from(e: CyclicDomainError) -> Self {
        Self::DomainError(e)
    }
}

impl From<super::lagrange_column::LagrangeColumnError> for UnivariateIopError {
    fn from(e: super::lagrange_column::LagrangeColumnError) -> Self {
        Self::LagrangeConstraintFailed(e)
    }
}
