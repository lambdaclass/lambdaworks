use lambdaworks_fft::errors::FFTError;
use lambdaworks_math::field::errors::FieldError;
use thiserror::Error;

use crate::{air::errors::AIRError, fri::errors::FriError};

#[derive(Debug, Error)]
pub enum StarkError {
    #[error("Attempt to preprocess randomized AIR using table with no columns")]
    RAPTraceColumns,
    #[error("Number of trace term gammas of DEEP composition polynomial should be {0} * {1} = {} but is {2}", .0 * .1)]
    DeepTraceTermGammas(usize, usize, usize),
    #[error("Could not get merkle proof of even composition polynomial")]
    CompositionPolyEvenProof,
    #[error("Number of evaluations of even composition polynomial should be {0} but number of LDE roots of unity is {1}")]
    CompositionPolyEvenEvaluations(usize, usize),
    #[error("Could not get merkle proof of odd composition polynomial")]
    CompositionPolyOddProof,
    #[error("Number of evaluations of odd composition polynomial should be {0} but number of LDE roots of unity is {1}")]
    CompositionPolyOddEvaluations(usize, usize),
    #[error("Could not get LDE trace merkle proof")]
    LDETraceMerkleProof,
    #[error("Could not get LDE trace row because index is {0} and number of trace rows is {1}")]
    LDETraceRowOutOfBounds(usize, usize),
    #[error("Number of AIR trace columns should be {0} but it's {0}")]
    AIRTraceColumns(usize, usize),
    #[error("AIR has no transition degrees")]
    AIRTransitionDegrees,
    #[error("AIR FRI transition queries field is set to zero")]
    AIRFriNumberOfQueries,
    #[error("Column index {0} is out of bounds for frame with {1} columns")]
    FrameColIndexOutOfBounds(usize, usize),
    #[error("Proof has no merkle roots of FRI layers")]
    ProofFriLayersMerkleRoots,
    #[error("Proof has no trace OOD frame evaluations")]
    ProofTraceFrameEvaluations,
    #[error("Could not compute trace term using offset {0} because zerofier is zero polynomial")]
    TraceTermZerofier(usize),
    #[error(
        "Could not evaluate boundary C_i because its zerofier evaluated at OOD point returned zero"
    )]
    BoundaryCiEvaluation,
    #[error(transparent)]
    AIR(#[from] AIRError),
    #[error(transparent)]
    FFT(#[from] FFTError),
    #[error(transparent)]
    Field(#[from] FieldError),
    #[error(transparent)]
    FRI(#[from] FriError),
}
