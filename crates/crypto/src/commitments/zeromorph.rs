//! Zeromorph PCS: reduces multilinear evaluation to univariate KZG opening.
//! Reference: Kohrita & Towa, "Zeromorph" (2023), https://eprint.iacr.org/2023/917

use crate::commitments::kzg::KateZaveruchaGoldberg;
use crate::commitments::multilinear::{IsMultilinearPCS, PcsError};
use crate::commitments::traits::IsCommitmentScheme;
use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::elliptic_curve::traits::IsPairing;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField, IsPrimeField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{AsBytes, ByteConversion};
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

/// Zeromorph PCS backed by KZG over an elliptic pairing.
#[derive(Clone)]
pub struct ZeromorphPCS<const N: usize, F, P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    P: IsPairing + Clone,
{
    kzg: KateZaveruchaGoldberg<F, P>,
}

/// Commitment for ZeromorphPCS: a single G1 point (the KZG commitment to f̂).
#[derive(Clone, Debug)]
pub struct ZeromorphCommitment<P: IsPairing> {
    pub g1: P::G1Point,
}

/// Opening proof for ZeromorphPCS.
#[derive(Clone, Debug)]
pub struct ZeromorphProof<const N: usize, F, P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    F::BaseType: Send + Sync,
    P: IsPairing,
{
    /// KZG commitments to each quotient polynomial q̂_k.
    pub q_commitments: Vec<P::G1Point>,
    /// Evaluation of f̂ at the random challenge x.
    pub z_f: FieldElement<F>,
    /// Evaluations of each q̂_k at the random challenge x.
    pub z_qs: Vec<FieldElement<F>>,
    /// Batched KZG opening proof.
    pub kzg_proof: P::G1Point,
}

impl<const N: usize, F, P> ZeromorphPCS<N, F, P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>>,
    P: IsPairing + Clone,
{
    pub fn new(kzg: KateZaveruchaGoldberg<F, P>) -> Self {
        Self { kzg }
    }
}

/// Compute quotient polynomials via the halving algorithm.
///
/// For a multilinear polynomial with evaluations `evals` and evaluation point
/// `point = (u_0, ..., u_{n-1})`, returns [q̂_0, ..., q̂_{n-1}] where q̂_k
/// has degree 2^k - 1.
///
/// The algorithm processes from k = n-1 downto 0, halving the current evaluation
/// vector at each step.
///
/// **Why Lagrange evaluations = univariate coefficients:** The Zeromorph protocol
/// uses the bijection between the boolean hypercube {0,1}^n and indices 0..2^n-1
/// to "unroll" a multilinear polynomial into a univariate: f̂(X) = Σ_i evals[i]·X^i.
/// The same convention applies to each quotient q̃_k — its boolean-hypercube
/// evaluations are used directly as the coefficients of q̂_k(X).
fn compute_zeromorph_quotients<F>(
    evals: &[FieldElement<F>],
    point: &[FieldElement<F>],
) -> Vec<Polynomial<FieldElement<F>>>
where
    F: IsField,
    F::BaseType: Send + Sync,
{
    let n = point.len();
    let mut current = evals.to_vec();
    let mut quotients = vec![Polynomial::zero(); n];

    for k in (0..n).rev() {
        let half = 1usize << k;
        // In lambdaworks, bit k of the eval index corresponds to variable r[n-1-k],
        // so we use point[n-1-k] (not point[k]) to match the paper's convention.
        let u_k = &point[n - 1 - k];
        let q_evals: Vec<FieldElement<F>> = (0..half)
            .map(|i| &current[i + half] - &current[i])
            .collect();
        let one_minus_u = FieldElement::<F>::one() - u_k;
        let new_current: Vec<FieldElement<F>> = (0..half)
            .map(|i| &one_minus_u * &current[i] + u_k * &current[i + half])
            .collect();
        current = new_current;
        quotients[k] = Polynomial::new(&q_evals);
    }
    debug_assert_eq!(
        current.len(),
        1,
        "halving loop should reduce to a single element (v)"
    );
    quotients
}

/// Compute Φ_m(y) = Π_{i=0}^{m-1}(1 + y^{2^i}).
///
/// This equals 1 + y + y^2 + ... + y^{2^m - 1}.
/// Φ_0(y) = 1 (empty product).
fn phi_eval<F: IsField>(y: &FieldElement<F>, m: usize) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
{
    if m == 0 {
        return FieldElement::one();
    }
    let mut result = FieldElement::<F>::one();
    let mut y_pow = y.clone();
    for _ in 0..m {
        result = &result * &(FieldElement::<F>::one() + &y_pow);
        y_pow = &y_pow * &y_pow;
    }
    result
}

/// Compute Ψ_{n,k}(x) = x^{2^k} * Φ_{n-k-1}(x^{2^{k+1}}) - u_k * Φ_{n-k}(x^{2^k}).
fn psi_eval<F: IsField>(
    x: &FieldElement<F>,
    u_k: &FieldElement<F>,
    n: usize,
    k: usize,
) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
{
    // x^{2^k} via k squarings
    let mut x_2k = x.clone();
    for _ in 0..k {
        x_2k = &x_2k * &x_2k;
    }
    // x^{2^{k+1}} = (x^{2^k})^2
    let x_2k1 = &x_2k * &x_2k;
    let phi_nk1 = phi_eval(&x_2k1, n - k - 1); // Φ_{n-k-1}(x^{2^{k+1}})
    let phi_nk = phi_eval(&x_2k, n - k); // Φ_{n-k}(x^{2^k})
    &x_2k * &phi_nk1 - u_k * &phi_nk
}

/// Derive internal Fiat-Shamir challenges for Zeromorph.
///
/// Absorbs the f commitment, evaluation point, claimed value, and quotient
/// commitments, then samples the evaluation challenge x and batching factor upsilon.
// TODO(security): The Zeromorph internal transcript is independent of any outer
// proof system transcript. In a standalone Spartan, the evaluation point r_y is
// derived from the Spartan transcript (which binds R1CS + public inputs + commitment),
// so proof transplant attacks are hard in practice. For recursive/IVC settings, the
// `IsMultilinearPCS::open` and `verify` signatures should be extended to accept an
// outer transcript binding (e.g., a hash of the current proof system transcript state).
fn derive_zeromorph_challenges<F, P>(
    f_commitment: &P::G1Point,
    point: &[FieldElement<F>],
    value: &FieldElement<F>,
    q_commitments: &[P::G1Point],
) -> (FieldElement<F>, FieldElement<F>)
where
    F: IsPrimeField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    P: IsPairing,
    P::G1Point: AsBytes,
{
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"zeromorph_v1");
    transcript.append_bytes(&f_commitment.as_bytes());
    transcript.append_bytes(&(point.len() as u64).to_be_bytes());
    for u_i in point {
        transcript.append_field_element(u_i);
    }
    transcript.append_field_element(value);
    transcript.append_bytes(&(q_commitments.len() as u64).to_be_bytes());
    for c in q_commitments {
        transcript.append_bytes(&c.as_bytes());
    }
    let x = transcript.sample_field_element();
    let upsilon = transcript.sample_field_element();
    (x, upsilon)
}

impl<const N: usize, F, P> IsMultilinearPCS<F> for ZeromorphPCS<N, F, P>
where
    F: IsPrimeField<CanonicalType = UnsignedInteger<N>> + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    P: IsPairing + Clone,
    P::G1Point: AsBytes + Clone,
{
    type Commitment = ZeromorphCommitment<P>;
    type Proof = ZeromorphProof<N, F, P>;
    type Error = PcsError;

    fn commit(
        &self,
        poly: &DenseMultilinearPolynomial<F>,
    ) -> Result<Self::Commitment, Self::Error> {
        let n_evals = poly.evals().len();
        if n_evals > self.kzg.srs_size() {
            return Err(PcsError(format!(
                "polynomial has {n_evals} evaluations but SRS only supports {}",
                self.kzg.srs_size()
            )));
        }
        let f_hat = Polynomial::new(poly.evals());
        let g1 = self.kzg.commit(&f_hat);
        Ok(ZeromorphCommitment { g1 })
    }

    fn open(
        &self,
        poly: &DenseMultilinearPolynomial<F>,
        point: &[FieldElement<F>],
    ) -> Result<(FieldElement<F>, Self::Proof), Self::Error> {
        let n = poly.num_vars();
        if point.len() != n {
            return Err(PcsError(format!(
                "point length {} != num_vars {}",
                point.len(),
                n
            )));
        }
        let n_evals = poly.evals().len();
        if n_evals > self.kzg.srs_size() {
            return Err(PcsError(format!(
                "polynomial has {n_evals} evaluations but SRS only supports {}",
                self.kzg.srs_size()
            )));
        }

        let value = poly
            .evaluate(point.to_vec())
            .map_err(|e| PcsError(format!("{e:?}")))?;

        let q_hats = compute_zeromorph_quotients(poly.evals(), point);

        let q_commitments: Vec<P::G1Point> = q_hats.iter().map(|q| self.kzg.commit(q)).collect();

        // Recompute f commitment for FS transcript (open() doesn't receive commitment)
        let f_hat = Polynomial::new(poly.evals());
        let f_commitment = self.kzg.commit(&f_hat);

        let (x, upsilon) =
            derive_zeromorph_challenges::<F, P>(&f_commitment, point, &value, &q_commitments);

        let z_f = f_hat.evaluate(&x);
        let z_qs: Vec<FieldElement<F>> = q_hats.iter().map(|q| q.evaluate(&x)).collect();

        let all_polys: Vec<Polynomial<FieldElement<F>>> =
            std::iter::once(f_hat).chain(q_hats).collect();
        let all_ys: Vec<FieldElement<F>> =
            std::iter::once(z_f.clone()).chain(z_qs.clone()).collect();
        let kzg_proof = self.kzg.open_batch(&x, &all_ys, &all_polys, &upsilon);

        Ok((
            value,
            ZeromorphProof {
                q_commitments,
                z_f,
                z_qs,
                kzg_proof,
            },
        ))
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &[FieldElement<F>],
        value: &FieldElement<F>,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        let n = point.len();

        let (x, upsilon) =
            derive_zeromorph_challenges::<F, P>(&commitment.g1, point, value, &proof.q_commitments);

        // Check Zeromorph identity: z_f - v*Φ_n(x) == Σ_k z_qs[k] * Ψ_{n,k}(x)
        let phi_n_x = phi_eval(&x, n);
        let lhs = &proof.z_f - value * &phi_n_x;
        // Use point[n-1-k] because in lambdaworks, bit k of the eval index corresponds
        // to variable r[n-1-k], matching the Zeromorph paper's variable convention.
        let rhs = proof
            .z_qs
            .iter()
            .enumerate()
            .map(|(k, z_k)| z_k * &psi_eval(&x, &point[n - 1 - k], n, k))
            .fold(FieldElement::<F>::zero(), |acc, t| acc + t);

        if lhs != rhs {
            return Ok(false);
        }

        // Batch KZG verify
        let all_commitments: Vec<P::G1Point> = std::iter::once(commitment.g1.clone())
            .chain(proof.q_commitments.clone())
            .collect();
        let all_ys: Vec<FieldElement<F>> = std::iter::once(proof.z_f.clone())
            .chain(proof.z_qs.clone())
            .collect();

        Ok(self
            .kzg
            .verify_batch(&x, &all_ys, &all_commitments, &proof.kzg_proof, &upsilon))
    }

    fn serialize_commitment(commitment: &Self::Commitment) -> Vec<u8> {
        commitment.g1.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitments::kzg::StructuredReferenceString;
    use lambdaworks_math::cyclic_group::IsGroup;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve,
        default_types::{FrElement, FrField},
        pairing::BLS12381AtePairing,
        twist::BLS12381TwistCurve,
    };
    use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
    use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

    type G1 = ShortWeierstrassJacobianPoint<BLS12381Curve>;
    type Kzg = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;
    type Zeromorph = ZeromorphPCS<4, FrField, BLS12381AtePairing>;
    type FE = FrElement;

    fn create_srs(
        size: usize,
    ) -> StructuredReferenceString<
        <BLS12381AtePairing as IsPairing>::G1Point,
        <BLS12381AtePairing as IsPairing>::G2Point,
    > {
        let toxic = FE::from(7u64);
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let powers: Vec<G1> = (0..size)
            .map(|i| g1.operate_with_self(toxic.pow(i as u128).canonical()))
            .collect();
        let g2_powers = [g2.clone(), g2.operate_with_self(toxic.canonical())];
        StructuredReferenceString::new(&powers, &g2_powers)
    }

    fn new_zeromorph(num_vars: usize) -> Zeromorph {
        let srs = create_srs(1 << num_vars);
        ZeromorphPCS::new(Kzg::new(srs))
    }

    #[test]
    fn test_zeromorph_undersized_srs_returns_error() {
        // 2-variable polynomial needs SRS of size >= 4, but we only give 2.
        let srs = create_srs(2); // undersized: only 2 powers
        let pcs = ZeromorphPCS::new(Kzg::new(srs));
        let evals = vec![FE::from(1u64), FE::from(2u64), FE::from(3u64), FE::from(4u64)];
        let poly = DenseMultilinearPolynomial::new(evals);
        let result = pcs.commit(&poly);
        assert!(result.is_err(), "commit with undersized SRS should return Err");
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("4") && msg.contains("2"), "error should mention sizes: {msg}");
    }

    #[test]
    fn test_zeromorph_pcs_1_var() {
        let pcs = new_zeromorph(1);
        let evals = vec![FE::from(3u64), FE::from(7u64)];
        let poly = DenseMultilinearPolynomial::new(evals);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(5u64)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok, "1-var Zeromorph proof should verify");
    }

    #[test]
    fn test_zeromorph_pcs_2_vars() {
        let pcs = new_zeromorph(2);
        let evals = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let poly = DenseMultilinearPolynomial::new(evals);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(2u64), FE::from(5u64)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok, "2-var Zeromorph proof should verify");
    }

    #[test]
    fn test_zeromorph_pcs_wrong_value() {
        let pcs = new_zeromorph(2);
        let evals = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let poly = DenseMultilinearPolynomial::new(evals);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(2u64), FE::from(5u64)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let wrong_value = value + FE::from(1u64);
        let ok = pcs
            .verify(&commitment, &point, &wrong_value, &proof)
            .unwrap();
        assert!(!ok, "Zeromorph should reject wrong value");
    }

    #[test]
    fn test_zeromorph_pcs_3_vars() {
        let pcs = new_zeromorph(3);
        let evals: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let poly = DenseMultilinearPolynomial::new(evals);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(2u64), FE::from(3u64), FE::from(4u64)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok, "3-var Zeromorph proof should verify");
    }

    #[test]
    fn test_zeromorph_pcs_tampered_commitment() {
        let pcs = new_zeromorph(2);
        let evals = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let poly = DenseMultilinearPolynomial::new(evals);
        let _commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(2u64), FE::from(5u64)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();

        let other_evals = vec![
            FE::from(9u64),
            FE::from(9u64),
            FE::from(9u64),
            FE::from(9u64),
        ];
        let other_poly = DenseMultilinearPolynomial::new(other_evals);
        let wrong_commitment = pcs.commit(&other_poly).unwrap();

        let ok = pcs
            .verify(&wrong_commitment, &point, &value, &proof)
            .unwrap();
        assert!(!ok, "Zeromorph should reject proof under wrong commitment");
    }

    #[test]
    fn test_zeromorph_pcs_constant_poly() {
        let pcs = new_zeromorph(2);
        let evals = vec![FE::from(42u64); 4];
        let poly = DenseMultilinearPolynomial::new(evals);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(3u64), FE::from(7u64)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        assert_eq!(value, FE::from(42u64));
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok, "Zeromorph should verify constant polynomial");
    }

    #[test]
    fn test_zeromorph_pcs_4_vars() {
        let pcs = new_zeromorph(4);
        let evals: Vec<FE> = (1u64..=16).map(FE::from).collect();
        let poly = DenseMultilinearPolynomial::new(evals);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
            FE::from(5u64),
        ];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok, "4-var Zeromorph proof should verify");
    }
}
