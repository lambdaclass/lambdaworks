pub mod prover;
pub mod verifier;

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;

use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::ByteConversion;

pub trait Channel<F: IsField> {
    fn append_field_element(&mut self, element: &FieldElement<F>);
    fn draw_felt(&mut self) -> FieldElement<F>;
}

impl<F> Channel<F> for DefaultTranscript<F>
where
    F: IsField,
    FieldElement<F>: ByteConversion,
{
    fn append_field_element(&mut self, element: &FieldElement<F>) {
        self.append_bytes(&element.to_bytes_be());
    }

    fn draw_felt(&mut self) -> FieldElement<F> {
        self.sample_field_element()
    }
}
