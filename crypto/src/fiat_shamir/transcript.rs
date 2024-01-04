use lambdaworks_math::{field::{traits::IsField, element::FieldElement}, polynomial::Polynomial, traits::ByteConversion};

pub trait Transcript {
    fn append(&mut self, new_data: &impl ToTranscript);
    fn challenge(&mut self) -> [u8; 32];
}

pub trait ToTranscript {
    fn to_transcript(&self) -> Vec<u8>;
}

impl ToTranscript for &[u8] {
    fn to_transcript(&self) -> Vec<u8> {
        self.to_vec()
    }
}

impl ToTranscript for Vec<u8> {
    fn to_transcript(&self) -> Vec<u8> {
        self.to_vec()
    }
}

impl ToTranscript for u64 {
    fn to_transcript(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl<F: IsField> ToTranscript for Polynomial<FieldElement<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> Vec<u8> {
        self.coefficients.to_transcript()
    }
}

impl<F: IsField> ToTranscript for FieldElement<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> Vec<u8> {
        self.value().to_bytes_le().to_vec()
    }
}

impl<F: IsField> ToTranscript for Vec<FieldElement<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn to_transcript(&self) -> Vec<u8> {
        self.iter()
            .fold(Vec::new(), |mut acc, val| {
                acc.extend_from_slice(&val.to_transcript());
                acc
            })
    }
}