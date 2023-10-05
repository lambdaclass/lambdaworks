use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

use crate::fiat_shamir::transcript::Transcript;

// Semantic Notes:
// - GOAL: Make the cleanest interface. ZK papers are inherently hard to read. This interface should strictly not cater to any one developers semantic beliefs but should cater to explicitly defining the PIOP interaction as succintly and explicitly as possible. Someone with just the proof scheme in front of them should be able to check the exchanged structs from this interface for a PCS scheme is correct in less than 15 min, less than 5 min if they know it well.
// - Since this is a PIOP we should use the Prover and Verifier naming convention arkworks adopts Committer and Verifier I think that makes it more confusing.
// - Setup and commit procedures sometime requires Rng Arkworks passes this this in. I think all params should be set by the PCS type and any external needed objects passed in with the PCS struct.
//   This maintains a clean trait interface that will contain the information explicitly defined in a protocol
//      - Shared objects between prover and verifier should be abstracted to the PCS struct??? -> Possible issues with encapsulation if we want the prover and verifier to be fully insolated. But since we are already viewing them both as one scheme I think this makes sense

pub trait IsCommitmentScheme<F: IsField> {
    //TODO: maybe make poly a type to abstract multilinear vs univariate vs bivariate??? Something to explore???
    //TODO: Maybe make public params part of struct since struct encompasses both prover and verifier
    type Proof;
    type Commitment;
    type PublicParams;
    type ProverKey;
    type VerifierKey;

    fn commit(
        &self,
        vk: &Self::VerifierKey,
        pp: &Self::PublicParams,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment;

    ///Note: maybe make this batch type an assc type. Motivation is maybe we want batches of proofs stored in different data structure depending on implementation
    /// TODO maybe make some of these optional
    fn open(
        &self,
        pk: &Self::ProverKey,
        pp: &Self::PublicParams,
        commitment: &Self::Commitment,
        poly: &Polynomial<FieldElement<F>>,
        point: &FieldElement<F>,
        transcript: &impl Transcript,
    ) -> Self::Commitment;

    fn open_batch(
        &self,
        pk: &Self::ProverKey,
        pp: &Self::PublicParams,
        commitments: &[Self::Commitment],
        polys: &[Polynomial<FieldElement<F>>],
        points: &[FieldElement<F>],
        transcript: &impl Transcript,
    ) -> Self::Commitment;

    fn verify(
        &self,
        vk: &Self::VerifierKey,
        pp: &Self::PublicParams,
        commitment: &Self::Commitment,
        point: &FieldElement<F>,
        eval: &FieldElement<F>,
        proof: &Self::Commitment,
        transcript: &impl Transcript,
    ) -> bool;

    fn verify_batch(
        &self,
        vk: &Self::VerifierKey,
        pp: &Self::PublicParams,
        commitments: &[Self::Commitment],
        commitment: &[Self::Commitment],
        points: &[FieldElement<F>],
        evals: &[FieldElement<F>],
        proofs: &[Self::Commitment],
        transcript: &impl Transcript,
    ) -> bool;
}

/*
pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment;

    fn open_batch(
        &self,
        x: &FieldElement<F>,
        y: &[FieldElement<F>],
        p: &[Polynomial<FieldElement<F>>],
        upsilon: &FieldElement<F>,
    ) -> Self::Commitment;

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool;

    fn verify_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Commitment,
        upsilon: &FieldElement<F>,
    ) -> bool;
}
*/
