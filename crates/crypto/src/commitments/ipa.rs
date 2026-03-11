use alloc::vec::Vec;
use core::marker::PhantomData;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    polynomial::Polynomial,
    traits::AsBytes,
    unsigned_integer::element::UnsignedInteger,
};

use crate::fiat_shamir::is_transcript::IsTranscript;

/// Transparent setup parameters for IPA.
///
/// Contains the public generators used by both prover and verifier.
/// No trusted setup is required — generators can be derived deterministically.
pub struct IpaSetup<G: IsGroup> {
    /// Pedersen-style generators G_0, ..., G_{n-1}
    pub generators: Vec<G>,
    /// Inner product binding point U
    pub u: G,
}

/// IPA proof structure.
///
/// Contains the left/right cross-term points from each folding round
/// and the final scalar after all rounds complete.
pub struct IpaProof<const N: usize, F: IsPrimeField, G: IsGroup> {
    /// Left cross-term points L_1, ..., L_k (one per round)
    pub l_points: Vec<G>,
    /// Right cross-term points R_1, ..., R_k (one per round)
    pub r_points: Vec<G>,
    /// Final scalar after log(n) rounds of folding
    pub a_final: FieldElement<F>,
}

/// Inner Product Argument polynomial commitment scheme.
///
/// Transparent (no trusted setup) PCS based on discrete log assumptions.
/// Produces O(log n) sized proofs with O(n) verification (dominated by a single MSM).
///
/// Generic over:
/// - `N`: number of limbs in the field's canonical representation
/// - `F`: scalar field (polynomial coefficients live here)
/// - `G`: group for commitments (must implement `AsBytes` for transcript serialization)
pub struct Ipa<const N: usize, F, G: IsGroup> {
    setup: IpaSetup<G>,
    _phantom: PhantomData<F>,
}

impl<const N: usize, F, G> Ipa<N, F, G>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    G: IsGroup + AsBytes,
{
    pub fn new(setup: IpaSetup<G>) -> Self {
        assert!(
            setup.generators.len().is_power_of_two(),
            "generator count must be a power of two"
        );
        Self {
            setup,
            _phantom: PhantomData,
        }
    }

    /// Commit to a polynomial: C = MSM(coefficients, generators).
    ///
    /// The polynomial is padded with zeros if its degree is less than n-1.
    pub fn commit(&self, p: &Polynomial<FieldElement<F>>) -> G {
        let n = self.setup.generators.len();
        let scalars = pad_coefficients::<N, F>(p, n);
        msm(&scalars, &self.setup.generators).expect("lengths match after padding")
    }

    /// Create an evaluation proof that p(z) = y.
    ///
    /// The transcript must be in the same state as the verifier's transcript
    /// will be at the start of verification.
    pub fn open(
        &self,
        p: &Polynomial<FieldElement<F>>,
        z: &FieldElement<F>,
        transcript: &mut impl IsTranscript<F>,
    ) -> IpaProof<N, F, G> {
        let n = self.setup.generators.len();
        let y = p.evaluate(z);

        // Compute commitment and bind it to the transcript
        let commitment = self.commit(p);
        seed_transcript(transcript, &commitment, z, &y);

        let mut a = pad_coefficients_fe::<F>(p, n);
        let mut b = compute_b_vector(z, n);
        let mut g = self.setup.generators.clone();

        let k = n.trailing_zeros() as usize; // log2(n)
        let mut l_points = Vec::with_capacity(k);
        let mut r_points = Vec::with_capacity(k);

        for _ in 0..k {
            let half = a.len() / 2;
            let (a_l, a_r) = a.split_at(half);
            let (b_l, b_r) = b.split_at(half);
            let (g_l, g_r) = g.split_at(half);

            // L_j = MSM(a_L, G_R) + <a_L, b_R> * U
            let l_msm = msm(&to_canonical(a_l), g_r).expect("lengths match");
            let l_ip = inner_product(a_l, b_r);
            let l_j = l_msm.operate_with(&self.setup.u.operate_with_self(l_ip.canonical()));

            // R_j = MSM(a_R, G_L) + <a_R, b_L> * U
            let r_msm = msm(&to_canonical(a_r), g_l).expect("lengths match");
            let r_ip = inner_product(a_r, b_l);
            let r_j = r_msm.operate_with(&self.setup.u.operate_with_self(r_ip.canonical()));

            // Append L, R to transcript and sample challenge
            transcript.append_bytes(&l_j.as_bytes());
            transcript.append_bytes(&r_j.as_bytes());
            let x: FieldElement<F> = transcript.sample_field_element();
            let x_inv = x.inv().expect("challenge is non-zero");

            l_points.push(l_j);
            r_points.push(r_j);

            // Fold: a' = x*a_L + x^{-1}*a_R
            //        b' = x^{-1}*b_L + x*b_R
            //        G' = x^{-1}*G_L + x*G_R
            let mut a_new = Vec::with_capacity(half);
            let mut b_new = Vec::with_capacity(half);
            let mut g_new = Vec::with_capacity(half);
            for i in 0..half {
                a_new.push(&x * &a_l[i] + &x_inv * &a_r[i]);
                b_new.push(&x_inv * &b_l[i] + &x * &b_r[i]);
                g_new.push(
                    g_l[i]
                        .operate_with_self(x_inv.canonical())
                        .operate_with(&g_r[i].operate_with_self(x.canonical())),
                );
            }
            a = a_new;
            b = b_new;
            g = g_new;
        }

        debug_assert_eq!(a.len(), 1);

        IpaProof {
            l_points,
            r_points,
            a_final: a.into_iter().next().unwrap(),
        }
    }

    /// Verify an evaluation proof.
    ///
    /// Returns true if the proof is valid for the claim that the polynomial
    /// committed to by `commitment` evaluates to `y` at point `z`.
    pub fn verify(
        &self,
        commitment: &G,
        z: &FieldElement<F>,
        y: &FieldElement<F>,
        proof: &IpaProof<N, F, G>,
        transcript: &mut impl IsTranscript<F>,
    ) -> bool {
        let n = self.setup.generators.len();
        let k = n.trailing_zeros() as usize;

        if proof.l_points.len() != k || proof.r_points.len() != k {
            return false;
        }

        // Reconstruct transcript state
        seed_transcript(transcript, commitment, z, y);

        // Collect challenges by replaying transcript
        let mut challenges = Vec::with_capacity(k);
        for j in 0..k {
            transcript.append_bytes(&proof.l_points[j].as_bytes());
            transcript.append_bytes(&proof.r_points[j].as_bytes());
            let x: FieldElement<F> = transcript.sample_field_element();
            challenges.push(x);
        }

        // P_final = P + Σ(x_j² * L_j + x_j^{-2} * R_j)
        // where P = commitment + <a, b> * U  but since we start from
        // P = MSM(a, G) + <a,b>*U, and commitment = MSM(a, G),
        // we need: P = commitment + y*U
        let mut p_final = commitment.operate_with(&self.setup.u.operate_with_self(y.canonical()));
        for (j, x) in challenges.iter().enumerate() {
            let x_sq = x * x;
            let x_inv = x.inv().expect("challenge is non-zero");
            let x_inv_sq = &x_inv * &x_inv;
            p_final = p_final
                .operate_with(&proof.l_points[j].operate_with_self(x_sq.canonical()))
                .operate_with(&proof.r_points[j].operate_with_self(x_inv_sq.canonical()));
        }

        // Compute s vector (tensor product of challenges)
        let s = compute_s_vector(&challenges);

        // G_final = MSM(s, G)
        let s_canonical: Vec<_> = s.iter().map(|si| si.canonical()).collect();
        let g_final = msm(&s_canonical, &self.setup.generators).expect("lengths match");

        // b_final via O(log n) computation
        let b_final = compute_b_final(z, &challenges);

        // Check: P_final == a_final * G_final + (a_final * b_final) * U
        let expected = g_final
            .operate_with_self(proof.a_final.canonical())
            .operate_with(
                &self
                    .setup
                    .u
                    .operate_with_self((&proof.a_final * &b_final).canonical()),
            );

        p_final == expected
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Dot product of two vectors of field elements.
fn inner_product<F: IsPrimeField>(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> FieldElement<F> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(FieldElement::zero(), |acc, (ai, bi)| acc + ai * bi)
}

/// Compute the evaluation vector b = [1, z, z², ..., z^{n-1}].
fn compute_b_vector<F: IsPrimeField>(z: &FieldElement<F>, n: usize) -> Vec<FieldElement<F>> {
    let mut b = Vec::with_capacity(n);
    let mut power = FieldElement::one();
    for _ in 0..n {
        b.push(power.clone());
        power = &power * z;
    }
    b
}

/// Compute the s vector as the recursive tensor product of challenges.
///
/// For challenges [x_1, ..., x_k], the s vector has 2^k entries where:
/// s_i = Π_{j=1}^{k} (if bit j of i is 1 then x_j else x_j^{-1})
///
/// This gives the coefficients for reconstructing G_final = MSM(s, G).
fn compute_s_vector<F: IsPrimeField>(challenges: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
    let k = challenges.len();
    let n = 1 << k;
    let mut s = Vec::with_capacity(n);
    s.push(FieldElement::one());

    // Build tensor product iteratively, processing challenges in reverse so that
    // challenges[0] (first round) corresponds to the most significant bit position.
    // This matches the folding convention: G' = x^{-1}*G_L + x*G_R where L = first half.
    for x in challenges.iter().rev() {
        let x_inv = x.inv().expect("challenge is non-zero");
        let current_len = s.len();
        // Extend with x * existing entries
        for i in 0..current_len {
            s.push(&s[i] * x);
        }
        // Multiply existing entries by x_inv
        for si in s.iter_mut().take(current_len) {
            *si = &*si * &x_inv;
        }
    }

    s
}

/// Compute b_final = <s, b> in O(log n) time.
///
/// b_final = Π_{j=1}^{k} (x_j^{-1} + x_j * z^{2^{k-j}})
fn compute_b_final<F: IsPrimeField>(
    z: &FieldElement<F>,
    challenges: &[FieldElement<F>],
) -> FieldElement<F> {
    let k = challenges.len();
    let mut result = FieldElement::one();

    // Precompute z powers: z^1, z^2, z^4, ..., z^{2^{k-1}}
    let mut z_powers = Vec::with_capacity(k);
    let mut zp = z.clone();
    for _ in 0..k {
        z_powers.push(zp.clone());
        zp = &zp * &zp;
    }

    for j in 0..k {
        let x_inv = challenges[j].inv().expect("challenge is non-zero");
        // z^{2^{k-1-j}} — index into z_powers reversed
        let z_pow = &z_powers[k - 1 - j];
        result = &result * &(&x_inv + &(&challenges[j] * z_pow));
    }

    result
}

/// Pad polynomial coefficients to length `n`, returning canonical representations.
fn pad_coefficients<const N: usize, F: IsPrimeField<CanonicalType = UnsignedInteger<N>>>(
    p: &Polynomial<FieldElement<F>>,
    n: usize,
) -> Vec<UnsignedInteger<N>> {
    let mut scalars: Vec<_> = p.coefficients.iter().map(|c| c.canonical()).collect();
    scalars.resize(n, UnsignedInteger::from_u64(0));
    scalars
}

/// Pad polynomial coefficients to length `n` as field elements.
fn pad_coefficients_fe<F: IsPrimeField>(
    p: &Polynomial<FieldElement<F>>,
    n: usize,
) -> Vec<FieldElement<F>> {
    let mut coeffs = p.coefficients.clone();
    coeffs.resize(n, FieldElement::zero());
    coeffs
}

/// Convert field elements to their canonical unsigned integer representation.
fn to_canonical<const N: usize, F: IsPrimeField<CanonicalType = UnsignedInteger<N>>>(
    elements: &[FieldElement<F>],
) -> Vec<UnsignedInteger<N>> {
    elements.iter().map(|e| e.canonical()).collect()
}

/// Seed the transcript with the commitment, evaluation point, and claimed value.
fn seed_transcript<F, G>(
    transcript: &mut impl IsTranscript<F>,
    commitment: &G,
    z: &FieldElement<F>,
    y: &FieldElement<F>,
) where
    F: IsPrimeField,
    G: IsGroup + AsBytes,
{
    transcript.append_bytes(b"ipa-v1");
    transcript.append_bytes(&commitment.as_bytes());
    transcript.append_field_element(z);
    transcript.append_field_element(y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::curves::pallas::curve::PallasCurve, traits::IsEllipticCurve,
        },
        field::fields::vesta_field::Vesta255PrimeField,
    };

    type F = Vesta255PrimeField;
    type FE = FieldElement<F>;
    type G = <PallasCurve as IsEllipticCurve>::PointRepresentation;

    /// Generate deterministic generators from the curve generator.
    fn test_setup(n: usize) -> IpaSetup<G> {
        let g = PallasCurve::generator();
        let generators: Vec<G> = (1..=n as u64).map(|i| g.operate_with_self(i)).collect();
        // U is derived from a different scalar to be independent of generators
        let u = g.operate_with_self(n as u64 + 1337);
        IpaSetup { generators, u }
    }

    fn make_transcript() -> DefaultTranscript<F> {
        DefaultTranscript::new(b"ipa-test")
    }

    #[test]
    fn inner_product_basic() {
        let a = [FE::from(2), FE::from(3), FE::from(4)];
        let b = [FE::from(5), FE::from(6), FE::from(7)];
        // 2*5 + 3*6 + 4*7 = 10 + 18 + 28 = 56
        assert_eq!(inner_product(&a, &b), FE::from(56));
    }

    #[test]
    fn inner_product_empty() {
        let a: [FE; 0] = [];
        let b: [FE; 0] = [];
        assert_eq!(inner_product(&a, &b), FE::zero());
    }

    #[test]
    fn commit_and_open_degree_0() {
        let setup = test_setup(1);
        let ipa = Ipa::<4, F, G>::new(setup);

        let p = Polynomial::new(&[FE::from(42)]);
        let z = FE::from(100);
        let y = p.evaluate(&z);
        assert_eq!(y, FE::from(42));

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());
        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn commit_and_open_degree_1() {
        let setup = test_setup(2);
        let ipa = Ipa::<4, F, G>::new(setup);

        // p(x) = 3 + 5x
        let p = Polynomial::new(&[FE::from(3), FE::from(5)]);
        let z = FE::from(7);
        let y = p.evaluate(&z); // 3 + 35 = 38

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());
        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn commit_and_open_degree_7() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());

        // 3 rounds for n=8
        assert_eq!(proof.l_points.len(), 3);
        assert_eq!(proof.r_points.len(), 3);

        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn commit_and_open_degree_15() {
        let setup = test_setup(16);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=16).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(5);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());

        // 4 rounds for n=16
        assert_eq!(proof.l_points.len(), 4);
        assert_eq!(proof.r_points.len(), 4);

        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn wrong_evaluation_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());

        // Claim wrong y
        let wrong_y = FE::from(9999);
        assert!(!ipa.verify(&commitment, &z, &wrong_y, &proof, &mut make_transcript()));
    }

    #[test]
    fn wrong_commitment_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let proof = ipa.open(&p, &z, &mut make_transcript());

        // Tamper with commitment
        let fake_commitment = PallasCurve::generator();
        assert!(!ipa.verify(&fake_commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn wrong_point_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());

        // Verify at a different point
        let wrong_z = FE::from(4);
        assert!(!ipa.verify(&commitment, &wrong_z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn s_vector_is_tensor_product() {
        // For challenges [x1, x2], s should be:
        // [x1^{-1}*x2^{-1}, x1^{-1}*x2, x1*x2^{-1}, x1*x2]
        let x1 = FE::from(3);
        let x2 = FE::from(7);
        let x1_inv = x1.inv().unwrap();
        let x2_inv = x2.inv().unwrap();

        let s = compute_s_vector(&[x1.clone(), x2.clone()]);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0], &x1_inv * &x2_inv);
        assert_eq!(s[1], &x1_inv * &x2);
        assert_eq!(s[2], &x1 * &x2_inv);
        assert_eq!(s[3], &x1 * &x2);
    }

    #[test]
    fn b_final_matches_inner_product() {
        // Verify that compute_b_final(z, challenges) == <s, b>
        let z = FE::from(5);
        let challenges = [FE::from(3), FE::from(7), FE::from(11)];

        let n = 1 << challenges.len(); // 8
        let b = compute_b_vector(&z, n);
        let s = compute_s_vector(&challenges);
        let ip = inner_product(&s, &b);

        let b_final = compute_b_final(&z, &challenges);
        assert_eq!(ip, b_final);
    }

    // -----------------------------------------------------------------------
    // Cross-validation tests against Python reference implementation
    // (ipa_reference.py using the same convention over Vesta scalar field)
    // -----------------------------------------------------------------------

    /// Helper: create a FieldElement from a hex string (big-endian).
    fn fe_from_hex(hex: &str) -> FE {
        FE::from_hex_unchecked(hex)
    }

    #[test]
    fn cross_validate_b_vector() {
        // Python: compute_b_vector(3, 8) = [1, 3, 9, 27, 81, 243, 729, 2187]
        let z = FE::from(3);
        let b = compute_b_vector(&z, 8);
        assert_eq!(b[0], FE::from(1));
        assert_eq!(b[1], FE::from(3));
        assert_eq!(b[2], FE::from(9));
        assert_eq!(b[3], FE::from(27));
        assert_eq!(b[4], FE::from(81));
        assert_eq!(b[5], FE::from(243));
        assert_eq!(b[6], FE::from(729));
        assert_eq!(b[7], FE::from(2187));
    }

    #[test]
    fn cross_validate_inner_product_polynomial() {
        // Python: p(3) = 1 + 2*3 + 3*9 + 4*27 + 5*81 + 6*243 + 7*729 + 8*2187 = 24604
        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let b = compute_b_vector(&FE::from(3), 8);
        let ip = inner_product(&coeffs, &b);
        assert_eq!(ip, FE::from(24604));
    }

    #[test]
    fn cross_validate_round1_inner_products() {
        // Python reference: Round 1 with a=[1..8], b=[1,3,9,27,81,243,729,2187]
        // <a_L, b_R> = 11502, <a_R, b_L> = 302
        let a: Vec<FE> = (1..=8).map(FE::from).collect();
        let b = compute_b_vector(&FE::from(3), 8);

        let (a_l, a_r) = a.split_at(4);
        let (b_l, b_r) = b.split_at(4);

        assert_eq!(inner_product(a_l, b_r), FE::from(11502));
        assert_eq!(inner_product(a_r, b_l), FE::from(302));
    }

    #[test]
    fn cross_validate_s_vector() {
        // Python: challenges = [11, 13, 17], s[7] = 2431
        // s[7] = x1 * x2 * x3 = 11 * 13 * 17 = 2431
        let challenges = [FE::from(11), FE::from(13), FE::from(17)];
        let s = compute_s_vector(&challenges);
        assert_eq!(s.len(), 8);

        // s[7] = x1 * x2 * x3 (all "R" branches)
        assert_eq!(s[7], FE::from(2431));

        // s[0] = x1^{-1} * x2^{-1} * x3^{-1} (all "L" branches)
        let x1_inv = FE::from(11).inv().unwrap();
        let x2_inv = FE::from(13).inv().unwrap();
        let x3_inv = FE::from(17).inv().unwrap();
        assert_eq!(s[0], &(&x1_inv * &x2_inv) * &x3_inv);

        // Cross-check with Python hex value for s[0]
        assert_eq!(
            s[0],
            fe_from_hex("30cf3e3c7226f6a0b299896cc8150fb0e9622cdac0db503d21298defe3256241")
        );
    }

    #[test]
    fn cross_validate_a_final() {
        // Python: after 3 rounds of folding a=[1..8] with challenges [11,13,17]:
        // a_final = 0x0ae6332dceed5c5a904514bed0b7a78aec097ed4e2fcc75ab2e67ba7e5ac6d1e
        let challenges = [FE::from(11), FE::from(13), FE::from(17)];
        let mut a: Vec<FE> = (1..=8).map(FE::from).collect();

        for x in &challenges {
            let half = a.len() / 2;
            let (a_l, a_r) = a.split_at(half);
            let x_inv = x.inv().unwrap();
            let a_new: Vec<FE> = (0..half).map(|i| x * &a_l[i] + &x_inv * &a_r[i]).collect();
            a = a_new;
        }

        assert_eq!(a.len(), 1);
        assert_eq!(
            a[0],
            fe_from_hex("0ae6332dceed5c5a904514bed0b7a78aec097ed4e2fcc75ab2e67ba7e5ac6d1e")
        );
    }

    #[test]
    fn cross_validate_b_final() {
        // Python: b_final with z=3, challenges=[11,13,17]:
        // 0x127b3563ef927b3563ef927b3563ef92851b3b3c75c2e2bb2f5a4c1cb24910f2
        let z = FE::from(3);
        let challenges = [FE::from(11), FE::from(13), FE::from(17)];
        let b_final = compute_b_final(&z, &challenges);

        assert_eq!(
            b_final,
            fe_from_hex("127b3563ef927b3563ef927b3563ef92851b3b3c75c2e2bb2f5a4c1cb24910f2")
        );

        // Also verify it matches <s, b>
        let b = compute_b_vector(&z, 8);
        let s = compute_s_vector(&challenges);
        assert_eq!(b_final, inner_product(&s, &b));
    }

    #[test]
    fn cross_validate_folding_preserves_inner_product() {
        // The key IPA invariant: <a', b'> = <a, b> + x^2*<a_L,b_R> + x^{-2}*<a_R,b_L>
        // (This is what makes L/R cross-terms correct)
        let a: Vec<FE> = (1..=8).map(FE::from).collect();
        let b = compute_b_vector(&FE::from(3), 8);
        let original_ip = inner_product(&a, &b);
        assert_eq!(original_ip, FE::from(24604));

        let x = FE::from(11);
        let x_inv = x.inv().unwrap();
        let half = 4;
        let (a_l, a_r) = a.split_at(half);
        let (b_l, b_r) = b.split_at(half);

        let l_ip = inner_product(a_l, b_r);
        let r_ip = inner_product(a_r, b_l);

        // Fold
        let a_new: Vec<FE> = (0..half).map(|i| &x * &a_l[i] + &x_inv * &a_r[i]).collect();
        let b_new: Vec<FE> = (0..half).map(|i| &x_inv * &b_l[i] + &x * &b_r[i]).collect();

        let new_ip = inner_product(&a_new, &b_new);

        // <a', b'> = <a_L, b_L> + <a_R, b_R> + x^2 * <a_L, b_R> + x^{-2} * <a_R, b_L>
        // = <a, b> + (x^2 - 1) * <a_L, b_R> + (x^{-2} - 1) * <a_R, b_L>
        // Wait, let's expand directly:
        // <a', b'> = Σ (x*a_L[i] + x^{-1}*a_R[i]) * (x^{-1}*b_L[i] + x*b_R[i])
        //          = Σ (a_L[i]*b_L[i] + x^2*a_L[i]*b_R[i] + x^{-2}*a_R[i]*b_L[i] + a_R[i]*b_R[i])
        //          = <a, b> + x^2 * L_ip + x^{-2} * R_ip
        // But <a, b> = <a_L, b_L> + <a_R, b_R> (since dot product is over full vectors)
        // Hmm, actually <a, b> = Σ a_i * b_i where first half uses (a_L, b_L) and second uses (a_R, b_R)
        // So <a, b> = <a_L, b_L> + <a_R, b_R>
        // And <a', b'> = <a_L, b_L> + <a_R, b_R> + x^2*<a_L, b_R> + x^{-2}*<a_R, b_L>
        //              = <a, b> + x^2 * L_ip + x^{-2} * R_ip

        let x_sq = &x * &x;
        let x_inv_sq = &x_inv * &x_inv;
        let expected = &original_ip + &(&x_sq * &l_ip) + &(&x_inv_sq * &r_ip);

        assert_eq!(new_ip, expected);
    }

    #[test]
    fn algebraic_invariant_every_round() {
        // Verify the full IPA invariant P = MSM(a, G) + <a, b> * U
        // holds at every round of the protocol.
        let setup = test_setup(8);
        let u = setup.u.clone();
        let original_generators = setup.generators.clone();
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        // Manually run the protocol and check invariant at each step
        let mut a = pad_coefficients_fe::<F>(&p, 8);
        let mut b = compute_b_vector(&z, 8);
        let mut g = original_generators.clone();

        // Initial invariant: P = MSM(a, G) + <a,b>*U = commitment + y*U
        let commitment = ipa.commit(&p);
        let p_point = commitment.operate_with(&u.operate_with_self(y.canonical()));

        // Verify initial invariant
        let a_canonical: Vec<_> = a.iter().map(|ai| ai.canonical()).collect();
        let msm_result = msm(&a_canonical, &g).unwrap();
        let ip = inner_product(&a, &b);
        let expected_p = msm_result.operate_with(&u.operate_with_self(ip.canonical()));
        assert_eq!(p_point, expected_p, "Initial invariant failed");

        // Use fixed challenges to make this deterministic
        let challenges = [FE::from(11), FE::from(13), FE::from(17)];

        let mut current_p = p_point;

        for (round, x) in challenges.iter().enumerate() {
            let half = a.len() / 2;
            let (a_l, a_r) = a.split_at(half);
            let (b_l, b_r) = b.split_at(half);
            let (g_l, g_r) = g.split_at(half);
            let x_inv = x.inv().unwrap();

            // Compute L and R
            let l_msm = msm(&to_canonical(a_l), g_r).unwrap();
            let l_ip = inner_product(a_l, b_r);
            let l_j = l_msm.operate_with(&u.operate_with_self(l_ip.canonical()));

            let r_msm = msm(&to_canonical(a_r), g_l).unwrap();
            let r_ip = inner_product(a_r, b_l);
            let r_j = r_msm.operate_with(&u.operate_with_self(r_ip.canonical()));

            // Fold
            let mut a_new = Vec::with_capacity(half);
            let mut b_new = Vec::with_capacity(half);
            let mut g_new = Vec::with_capacity(half);
            for i in 0..half {
                a_new.push(x * &a_l[i] + &x_inv * &a_r[i]);
                b_new.push(&x_inv * &b_l[i] + x * &b_r[i]);
                g_new.push(
                    g_l[i]
                        .operate_with_self(x_inv.canonical())
                        .operate_with(&g_r[i].operate_with_self(x.canonical())),
                );
            }

            // Update P: P' = P + x^2 * L + x^{-2} * R
            let x_sq = x * x;
            let x_inv_sq = &x_inv * &x_inv;
            current_p = current_p
                .operate_with(&l_j.operate_with_self(x_sq.canonical()))
                .operate_with(&r_j.operate_with_self(x_inv_sq.canonical()));

            a = a_new;
            b = b_new;
            g = g_new;

            // Verify invariant: P' = MSM(a', G') + <a', b'> * U
            let a_can: Vec<_> = a.iter().map(|ai| ai.canonical()).collect();
            let msm_result = msm(&a_can, &g).unwrap();
            let ip = inner_product(&a, &b);
            let expected = msm_result.operate_with(&u.operate_with_self(ip.canonical()));
            assert_eq!(
                current_p,
                expected,
                "Invariant failed at round {}",
                round + 1
            );
        }

        // After all rounds, a and b should be single elements
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);

        // Final check: a_final matches Python reference
        assert_eq!(
            a[0],
            fe_from_hex("0ae6332dceed5c5a904514bed0b7a78aec097ed4e2fcc75ab2e67ba7e5ac6d1e")
        );
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn zero_polynomial() {
        let setup = test_setup(4);
        let ipa = Ipa::<4, F, G>::new(setup);

        // p(x) = 0
        let p = Polynomial::new(&[FE::zero()]);
        let z = FE::from(42);
        let y = p.evaluate(&z);
        assert_eq!(y, FE::zero());

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());
        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn evaluate_at_zero() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        // p(x) = 3 + 5x + 7x^2, evaluated at z=0 should give 3
        let p = Polynomial::new(&[FE::from(3), FE::from(5), FE::from(7)]);
        let z = FE::zero();
        let y = p.evaluate(&z);
        assert_eq!(y, FE::from(3));

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());
        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn polynomial_degree_less_than_n() {
        // Degree-2 polynomial with n=8 generators (needs padding)
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let p = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);
        let z = FE::from(10);
        let y = p.evaluate(&z); // 1 + 20 + 300 = 321

        let commitment = ipa.commit(&p);
        let proof = ipa.open(&p, &z, &mut make_transcript());
        assert!(ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn tampered_l_point_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let mut proof = ipa.open(&p, &z, &mut make_transcript());

        // Tamper with the first L point
        proof.l_points[0] = PallasCurve::generator();
        assert!(!ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn tampered_r_point_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let mut proof = ipa.open(&p, &z, &mut make_transcript());

        // Tamper with the last R point
        let last = proof.r_points.len() - 1;
        proof.r_points[last] = PallasCurve::generator();
        assert!(!ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn tampered_a_final_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let mut proof = ipa.open(&p, &z, &mut make_transcript());

        // Tamper with a_final
        proof.a_final += FE::one();
        assert!(!ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }

    #[test]
    fn wrong_proof_length_rejected() {
        let setup = test_setup(8);
        let ipa = Ipa::<4, F, G>::new(setup);

        let coeffs: Vec<FE> = (1..=8).map(FE::from).collect();
        let p = Polynomial::new(&coeffs);
        let z = FE::from(3);
        let y = p.evaluate(&z);

        let commitment = ipa.commit(&p);
        let mut proof = ipa.open(&p, &z, &mut make_transcript());

        // Remove one L point to create a length mismatch
        proof.l_points.pop();
        assert!(!ipa.verify(&commitment, &z, &y, &proof, &mut make_transcript()));
    }
}
