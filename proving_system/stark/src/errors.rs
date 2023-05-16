use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

use crate::{air::errors::AIRError, fri::errors::FriError};

#[derive(Debug, Error)]
pub enum StarkError {
    #[error("Could not evaluate polynomial: {0}")]
    PolynomialEvaluationError(FFTError),
    #[error("Number of trace term gammas of DEEP composition polynomial should be {0} * {1} = {} but is {2}", .0 * .1)]
    DeepTraceTermGammasError(usize, usize, usize),
    #[error("Could not get merkle proof of even composition polynomial")]
    CompositionPolyEvenProofError,
    #[error("Number of evaluations of even composition polynomial should be {0} but number of LDE roots of unity is {1}")]
    CompositionPolyEvenEvaluationsError(usize, usize),
    #[error("Could not get merkle proof of odd composition polynomial")]
    CompositionPolyOddProofError,
    #[error("Number of evaluations of odd composition polynomial should be {0} but number of LDE roots of unity is {1}")]
    CompositionPolyOddEvaluationsError(usize, usize),
    #[error("Could not get LDE trace merkle proof")]
    LDETraceMerkleProofError,
    #[error("Could not create domain: {0}")]
    DomainCreationError(FFTError),
    #[error("Could not query FRI layers: {0}")]
    FriQueryError(FriError),
    #[error("AIR trace columns field is set to zero")]
    AIRTraceColumnsError,
    #[error("AIR has no transition degrees")]
    AIRTransitionDegreesError,
    #[error("Could not verify composition polynomial: {0}")]
    CompositionPolyVerificationError(AIRError),
    #[error("Could not reconstruct DEEP composition polynomial: {0}")]
    DeepPolyReconstructionError(FFTError),
    #[error("Could not verify query: {0}")]
    QueryVerificationError(FFTError),
    #[error("Could not evaluate constraints: {0}")]
    ConstraintEvaluationError(AIRError),
}
