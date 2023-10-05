use crate::{fiat_shamir::transcript::Transcript, merkle_tree::proof::Proof};

use super::traits::IsCommitmentScheme;
use digest::Digest;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    polynomial::Polynomial,
    traits::{Deserializable, Serializable},
    unsigned_integer::element::UnsignedInteger,
};
use rand::prelude::*;

pub struct Key<G: IsGroup> {
    /// commitment key: trimmed vector of generator points from a random generator. Used to commit the coeffs to.
    pub ck: Vec<G>,
    /// H: group element used as generator used to create commitment key
    pub h: G,
    /// s: group element used as generator for hiding -> Maybe make into an option
    pub s: G,
    /// maximum degree supported by the parameters key derived from setup()
    pub max_degree: usize,
}

///NOTE: Check paper and abstractions to see if these should be distributed amongst prover and verifier keys.
/// We should make adding randomness/hiding/zk optional as in arkworks -> Wrap in option.
pub struct PublicParameters<G: IsGroup> {
    /// sigma_pows: set G1 elements of which pk and vk and grabbed from as truncated subsets
    pub sigma_powers: Vec<G>,
    /// H: group element used as generator used to create commitment key
    pub h: G,
    /// s: group element used as generator for hiding -> Maybe make into an option
    pub s: G,
    /// maximum degree supported by the parameters key derived from setup()
    pub max_degree: usize,
}

pub struct IPAProof<G: IsGroup> {
    pub l_vec: Vec<G>,
    pub r_vec: Vec<G>,
    /// Last Commitment key of last opening round
    pub u: G,
    /// Last Commitment input for last opening round
    pub c: G,
    /// Hiding Commitment to p
    pub hiding_c: G,
    /// Hiding commitment randomness
    pub w: G,
}

#[derive(Debug, Default)]
pub struct IPAPolynomialCommitment<'a, R, D, G>
where
    &'a mut R: RngCore + Default + Copy,
    G: IsGroup,
    D: Digest,
{
    pub rng: &'a mut R,
    _digest: D,
    _phantom: G,
}

impl<'a, R, D, G> IPAPolynomialCommitment<'a, R, D, G>
where
    &'a mut R: RngCore + Default + Copy,
    G: IsGroup + Deserializable + Serializable,
    D: Digest,
{
    ///Setup function seed
    pub const PROTOCOL_NAME: &'static [u8] = b"IPA PCS 2023";

    ///TODO: maybe make this new? Maybe a trait method???
    pub fn setup(&self, max_degree: usize) -> PublicParameters<G> {
        // Ensure max_degree is a power of two -> TODO maybe make this an error instead of rounding up???
        assert!((max_degree + 1).is_power_of_two());

        // we take the last two generators as h and s -> (max_degree + 1) + 2
        //TODO: ALl this just to generate random scalars common gotta be a better way.
        let mut sigma_powers: Vec<_> = (0..(max_degree + 3))
            .into_iter()
            .map(|i| {
                //For now we use a default hasher as we don't support hash to curve
                let mut hash =
                    D::digest([Self::PROTOCOL_NAME, &i.to_le_bytes()].concat().as_slice());
                let mut g = G::deserialize(&hash);
                let mut j = 0u64;
                //If hash of name and iteration doesn't work keep geussing till it does lol this is from arkworks!!!!
                while g.is_err() {
                    // PROTOCOL NAME, i, j
                    let mut bytes = Self::PROTOCOL_NAME.to_vec();
                    bytes.extend(i.to_le_bytes());
                    bytes.extend(j.to_le_bytes());
                    hash = D::digest(bytes.as_slice());
                    g = G::deserialize(&hash);
                    j += 1;
                }
                //TODO: Do we need to convert this to affine???
                g.unwrap()
            })
            .collect();

        let h = sigma_powers.pop().unwrap();
        let s = sigma_powers.pop().unwrap();

        //TODO: maybe make this a result???
        PublicParameters {
            sigma_powers,
            h,
            s,
            max_degree,
        }
    }

    pub fn trim(&self, pp: PublicParameters<G>, d: usize) -> (Key<G>, Key<G>) {
        //TODO: make this a trimming degree to large error
        assert!((d + 1) < pp.max_degree);
        assert!((d + 1).is_power_of_two());

        //(vk, pk)
        (
            Key {
                ck: pp.sigma_powers[0..(d + 1)].to_vec(),
                h: pp.h.clone(),
                s: pp.h.clone(),
                max_degree: pp.max_degree,
            },
            Key {
                ck: pp.sigma_powers[0..(d + 1)].to_vec(),
                h: pp.h.clone(),
                s: pp.h.clone(),
                max_degree: pp.max_degree,
            },
        )
    }

    pub fn succint_check(
        &self,
        vk: &Key<G>,
        commitment: &G,
        point: &G,
        eval: &G,
        proof: &IPAProof<G>,
        transcript: &impl Transcript,
        degree: usize,
    ) -> (Polynomial<G>, G) {
        let log_d = log2(vk.max_degree + 1) as usize;

        assert!(degree == vk.max_degree);
        //TODO: toggle this for non-hiding
        let alpha = transcript.append(&commitment.serialize());
        todo!();
    }
}

fn log2(x: usize) -> u32 {
    if x == 0 {
        0
    } else if x.is_power_of_two() {
        1usize.leading_zeros() - x.leading_zeros()
    } else {
        0usize.leading_zeros() - x.leading_zeros()
    }
}
//Note: Arkworks decided to create three structs to specify a commitment of a Group element. I think thats bad abstraction but the author has a phd so what do I know. Where just going to use a FieldElement<F> b/c apparently I must be small brained.

impl<'a, const N: usize, F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>, G, D, R>
    IsCommitmentScheme<F> for IPAPolynomialCommitment<'a, R, D, G>
where
    &'a mut R: RngCore + Default + Copy,
    G: IsGroup,
    D: Digest,
{
    type Commitment = G;
    type Proof = IPAProof<G>;
    type PublicParams = PublicParameters<G>;
    type ProverKey = Key<G>;
    type VerifierKey = Key<G>;

    // commit(): https://github.com/arkworks-rs/poly-commit/blob/master/src/ipa_pc/mod.rs#L416
    // TODO: generalize for many commits
    fn commit(
        &self,
        vk: &Self::VerifierKey,
        pp: &Self::PublicParams,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment {
        let coefficients: Vec<_> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        msm(&coefficients, &vk.ck).expect("`points` is sliced by `cs`'s length")
    }

    // open(): https://github.com/arkworks-rs/poly-commit/blob/master/src/ipa_pc/mod.rs#L488
    fn open(
        &self,
        pk: &Self::ProverKey,
        pp: &Self::PublicParams,
        commitment: &Self::Commitment,
        poly: &Polynomial<FieldElement<F>>,
        point: &FieldElement<F>,
        transcript: &impl Transcript,
    ) -> Self::Commitment {
        todo!()
    }

    //open_combinations(): https://github.com/arkworks-rs/poly-commit/blob/master/src/ipa_pc/mod.rs#L875
    fn open_batch(
        &self,
        pk: &Self::ProverKey,
        pp: &Self::PublicParams,
        commitment: &[Self::Commitment],
        poly: &[Polynomial<FieldElement<F>>],
        point: &[FieldElement<F>],
        transcript: &impl Transcript,
    ) -> Self::Commitment {
        todo!()
    }

    fn verify(
        &self,
        vk: &Self::VerifierKey,
        pp: &Self::PublicParams,
        commitment: &Self::Commitment,
        point: &FieldElement<F>,
        eval: &FieldElement<F>,
        proof: &Self::Commitment,
        transcript: &impl Transcript,
    ) -> bool {
        todo!()
    }

    // check_combinations(): https://github.com/arkworks-rs/poly-commit/blob/master/src/ipa_pc/mod.rs#L985
    // &
    // batch_check(): https://github.com/arkworks-rs/poly-commit/blob/master/src/ipa_pc/mod.rs#L789
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
    ) -> bool {
        todo!()
    }
}
