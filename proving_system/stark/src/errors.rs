use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

use crate::{air::errors::AIRError, fri::errors::FriError};

#[derive(Debug, Error)]
pub enum StarkError {
    #[error("Could not evaluate polynomial: {0}")]
    PolynomialEvaluation(FFTError),
    #[error("Could not interpolate and commit: {0}")]
    InterpolationAndCommitment(AIRError),
    #[error("Could not preprocess randomized AIR: {0}")]
    RandomizedAIRPreprocessing(AIRError),
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
    #[error("Could not create domain: {0}")]
    DomainCreation(FFTError),
    #[error("Could not query FRI layers: {0}")]
    FriQuery(FriError),
    #[error("Number of AIR trace columns should be {0} but it's {0}")]
    AIRTraceColumns(usize, usize),
    #[error("AIR has no transition degrees")]
    AIRTransitionDegrees,
    #[error("AIR FRI transition queries field is set to zero")]
    AIRFriNumberOfQueries,
    #[error("Could not verify composition polynomial: {0}")]
    CompositionPolyVerification(AIRError),
    #[error("Could not reconstruct DEEP composition polynomial: {0}")]
    DeepPolyReconstruction(FFTError),
    #[error("Could not verify query: {0}")]
    QueryVerification(FFTError),
    #[error("Could not evaluate constraints: {0}")]
    ConstraintEvaluation(AIRError),
    #[error("Column index {0} is out of bounds for frame with {1} columns")]
    FrameColIndexOutOfBounds(usize, usize),
    #[error("Proof has no merkle roots of FRI layers")]
    ProofFriLayersMerkleRoots,
    #[error("Proof has no trace OOD frame evaluations")]
    ProofTraceFrameEvaluations,
}
