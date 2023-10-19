use crate::prover::Proof;
use crate::setup::VerifyingKey;

#[derive(Default)]
pub struct Verifier {}

impl Verifier {
    // todo! check if args should be extended (e.g. public_input)
    pub fn verify(&self, _p: &Proof, _vk: &VerifyingKey) -> bool {
        // todo!
        true
    }
}

#[cfg(test)]
mod tests {
    // todo!
}
