use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};
use sha3::{Digest, Keccak256};

/// The functionality of a transcript to be used in the STARK Prove and Verify protocols.
pub trait IsTranscript<F: IsField> {
    /// Appends a field element to the transcript.
    fn append_field_element(&mut self, element: &FieldElement<F>);
    /// Appends a bytes to the transcript.
    fn append_bytes(&mut self, new_bytes: &[u8]);
    /// Returns the inner state of the transcript that fully determines its outputs.
    fn state(&self) -> [u8; 32];
    /// Returns a random field element.
    fn sample_field_element(&mut self) -> FieldElement<F>;
    /// Returns a random index between 0 and `upper_bound`.
    fn sample_u64(&mut self, upper_bound: u64) -> u64;
    /// Returns a field element not contained in `lde_roots_of_unity_coset` or `trace_roots_of_unity`.
    fn sample_z_ood<S: IsSubFieldOf<F>>(
        &mut self,
        lde_roots_of_unity_coset: &[FieldElement<S>],
        trace_roots_of_unity: &[FieldElement<S>],
    ) -> FieldElement<F>
    where
        FieldElement<F>: AsBytes,
    {
        loop {
            let value: FieldElement<F> = self.sample_field_element();
            if !lde_roots_of_unity_coset
                .iter()
                .any(|x| x.clone().to_extension() == value)
                && !trace_roots_of_unity
                    .iter()
                    .any(|x| x.clone().to_extension() == value)
            {
                return value;
            }
        }
    }

    fn keccak_hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(data);
        let mut result_hash = [0_u8; 32];
        result_hash.copy_from_slice(&hasher.finalize_reset());
        result_hash
    }
}
