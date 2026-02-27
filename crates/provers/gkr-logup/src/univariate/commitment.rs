use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment(pub Vec<u8>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningProof(pub Vec<u8>);

pub trait PolynomialCommitment<F: IsField> {
    fn commit(polynomial: &[FieldElement<F>]) -> Commitment;

    fn open(
        polynomial: &[FieldElement<F>],
        point: &FieldElement<F>,
        value: &FieldElement<F>,
    ) -> Result<OpeningProof, CommitmentError>;

    fn verify(
        commit: &Commitment,
        point: &FieldElement<F>,
        value: &FieldElement<F>,
        proof: &OpeningProof,
    ) -> Result<bool, CommitmentError>;
}

#[derive(Debug, Clone)]
pub enum CommitmentError {
    InvalidInput(String),
    VerificationFailed,
    NotImplemented,
}

impl core::fmt::Display for CommitmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CommitmentError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            CommitmentError::VerificationFailed => write!(f, "Verification failed"),
            CommitmentError::NotImplemented => write!(f, "Not implemented"),
        }
    }
}

impl core::error::Error for CommitmentError {}

pub struct DummyCommitmentScheme<F: IsField> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: IsField> PolynomialCommitment<F> for DummyCommitmentScheme<F> {
    fn commit(_polynomial: &[FieldElement<F>]) -> Commitment {
        Commitment(vec![])
    }

    fn open(
        _polynomial: &[FieldElement<F>],
        _point: &FieldElement<F>,
        _value: &FieldElement<F>,
    ) -> Result<OpeningProof, CommitmentError> {
        Ok(OpeningProof(vec![]))
    }

    fn verify(
        _commit: &Commitment,
        _point: &FieldElement<F>,
        _value: &FieldElement<F>,
        _proof: &OpeningProof,
    ) -> Result<bool, CommitmentError> {
        Ok(true)
    }
}

pub struct KZG10Commitment<F: IsField> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: IsField> PolynomialCommitment<F> for KZG10Commitment<F> {
    fn commit(_polynomial: &[FieldElement<F>]) -> Commitment {
        unimplemented!("KZG10 commitment requires trusted setup and pairing support")
    }

    fn open(
        _polynomial: &[FieldElement<F>],
        _point: &FieldElement<F>,
        _value: &FieldElement<F>,
    ) -> Result<OpeningProof, CommitmentError> {
        unimplemented!("KZG10 opening requires pairing support")
    }

    fn verify(
        _commit: &Commitment,
        _point: &FieldElement<F>,
        _value: &FieldElement<F>,
        _proof: &OpeningProof,
    ) -> Result<bool, CommitmentError> {
        unimplemented!("KZG10 verification requires pairing support")
    }
}
