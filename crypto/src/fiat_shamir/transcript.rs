use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::{
        dense_multilinear_poly::DenseMultilinearPolynomial,
        sparse_multilinear_poly::SparseMultilinearPolynomial,
    },
    traits::ByteConversion,
};

pub trait Transcript {
    fn append(&mut self, new_data: &[u8]);

    fn challenge(&mut self) -> [u8; 32];
}

pub trait ToTranscript {
    fn to_transcript(&self) -> &[u8];
}

impl ToTranscript for &[u8] {
    fn to_transcript(&self) -> &[u8] {
        &self
    }
}

impl ToTranscript for Vec<u8> {
    fn to_transcript(&self) -> &[u8] {
        &self.as_slice()
    }
}

impl ToTranscript for u64 {
    fn to_transcript(&self) -> &[u8] {
        &self.to_le_bytes()
    }
}

impl<F: IsField> ToTranscript for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> &[u8] {
        self.evals()
            .iter()
            .fold(Vec::new(), |mut acc, val| {
                acc.extend_from_slice(val.to_transcript());
                acc
            })
            .as_slice()
    }
}

impl<F: IsField> ToTranscript for SparseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> &[u8] {
        self.evals()
            .iter()
            .fold(Vec::new(), |mut acc, (i, val)| {
                acc.extend_from_slice(val.to_transcript());
                acc
            })
            .as_slice()
    }
}

impl<F: IsField> ToTranscript for FieldElement<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> &[u8] {
        &self.value().to_bytes_le()
    }
}

impl<F: IsField> ToTranscript for Vec<FieldElement<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> &[u8] {
        self.iter()
            .fold(Vec::new(), |mut acc, val| {
                acc.extend_from_slice(val.to_transcript());
                acc
            })
            .as_slice()
    }
}
