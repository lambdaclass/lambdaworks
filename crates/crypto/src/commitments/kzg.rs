use super::traits::IsCommitmentScheme;
use alloc::{borrow::ToOwned, vec::Vec};
use core::{marker::PhantomData, mem};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::IsPairing,
    errors::DeserializationError,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    polynomial::Polynomial,
    traits::{AsBytes, Deserializable},
    unsigned_integer::element::UnsignedInteger,
};

#[derive(PartialEq, Clone, Debug)]
pub struct StructuredReferenceString<G1Point, G2Point> {
    /// Vector of points in G1 encoding g1, s g1, s^2 g1, s^3 g1, ... s^n g1
    pub powers_main_group: Vec<G1Point>,
    /// Slice of points in G2 encoding g2, s g2
    /// We could relax this to include more powers, but for most applications
    /// this suffices
    pub powers_secondary_group: [G2Point; 2],
}

impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup,
    G2Point: IsGroup,
{
    /// Creates a new SRS from slices of G1points and a slice of length 2 of G2 points
    pub fn new(powers_main_group: &[G1Point], powers_secondary_group: &[G2Point; 2]) -> Self {
        Self {
            powers_main_group: powers_main_group.into(),
            powers_secondary_group: powers_secondary_group.clone(),
        }
    }
}

#[cfg(feature = "std")]
impl<G1Point, G2Point> StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + Deserializable,
    G2Point: IsGroup + Deserializable,
{
    /// Read SRS from file
    pub fn from_file(file_path: &str) -> Result<Self, crate::errors::SrsFromFileError> {
        let bytes = std::fs::read(file_path)?;
        Ok(Self::deserialize(&bytes)?)
    }
}

impl<G1Point, G2Point> AsBytes for StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + AsBytes,
    G2Point: IsGroup + AsBytes,
{
    /// Serialize SRS
    fn as_bytes(&self) -> Vec<u8> {
        let mut serialized_data: Vec<u8> = Vec::new();
        // First 4 bytes encodes protocol version
        let protocol_version: [u8; 4] = [0; 4];

        serialized_data.extend(&protocol_version);

        // Second 8 bytes store the amount of G1 elements to be stored, this is more than can be indexed with a 64-bit architecture, and some millions of terabytes of data if the points were compressed
        let mut main_group_len_bytes: Vec<u8> = self.powers_main_group.len().to_le_bytes().to_vec();

        // For architectures with less than 64 bits for pointers
        // We add extra zeros at the end
        while main_group_len_bytes.len() < 8 {
            main_group_len_bytes.push(0)
        }

        serialized_data.extend(&main_group_len_bytes);

        // G1 elements
        for point in &self.powers_main_group {
            serialized_data.extend(point.as_bytes());
        }

        // G2 elements
        for point in &self.powers_secondary_group {
            serialized_data.extend(point.as_bytes());
        }

        serialized_data
    }
}

impl<G1Point, G2Point> Deserializable for StructuredReferenceString<G1Point, G2Point>
where
    G1Point: IsGroup + Deserializable,
    G2Point: IsGroup + Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
        const MAIN_GROUP_LEN_OFFSET: usize = 4;
        const MAIN_GROUP_OFFSET: usize = 12;

        let main_group_len_u64 = u64::from_le_bytes(
            bytes
                .get(MAIN_GROUP_LEN_OFFSET..MAIN_GROUP_OFFSET)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .expect("slice length is exactly 8 bytes for u64"),
        );

        let main_group_len = usize::try_from(main_group_len_u64)
            .map_err(|_| DeserializationError::PointerSizeError)?;

        let mut main_group: Vec<G1Point> = Vec::new();
        let mut secondary_group: Vec<G2Point> = Vec::new();

        let size_g1_point = mem::size_of::<G1Point>();
        let size_g2_point = mem::size_of::<G2Point>();

        for i in 0..main_group_len {
            let start = i * size_g1_point + MAIN_GROUP_OFFSET;
            let end = start + size_g1_point;
            let point_bytes = bytes
                .get(start..end)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?;
            let point = G1Point::deserialize(point_bytes)?;
            main_group.push(point);
        }

        let g2s_offset = size_g1_point * main_group_len + 12;
        for i in 0..2 {
            let start = i * size_g2_point + g2s_offset;
            let end = start + size_g2_point;
            let point_bytes = bytes
                .get(start..end)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?;
            let point = G2Point::deserialize(point_bytes)?;
            secondary_group.push(point);
        }

        let secondary_group_slice = [secondary_group[0].clone(), secondary_group[1].clone()];

        let srs = StructuredReferenceString::new(&main_group, &secondary_group_slice);
        Ok(srs)
    }
}

#[derive(Clone)]
pub struct KateZaveruchaGoldberg<F: IsPrimeField, P: IsPairing> {
    srs: StructuredReferenceString<P::G1Point, P::G2Point>,
    phantom: PhantomData<F>,
}

impl<F: IsPrimeField, P: IsPairing> KateZaveruchaGoldberg<F, P> {
    pub fn new(srs: StructuredReferenceString<P::G1Point, P::G2Point>) -> Self {
        Self {
            srs,
            phantom: PhantomData,
        }
    }
}

impl<const N: usize, F: IsPrimeField<CanonicalType = UnsignedInteger<N>>, P: IsPairing>
    IsCommitmentScheme<F> for KateZaveruchaGoldberg<F, P>
{
    type Commitment = P::G1Point;

    /// Given a polynomial and an SRS, creates a commitment to p(x), which corresponds to a G1 point
    /// The commitment is p(s) g1, evaluated as \sum_i c_i srs.powers_main_group[i], where c_i are the coefficients
    /// of the polynomial.
    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment {
        let coefficients: Vec<_> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.canonical())
            .collect();
        msm(
            &coefficients,
            &self.srs.powers_main_group[..coefficients.len()],
        )
        .expect("`points` is sliced by `cs`'s length")
    }

    /// Creates an evaluation proof for the polynomial p at x equal to y.
    /// This is a commitment to the quotient polynomial q(t) = (p(t) - y)/(t - x)
    /// The commitment is simply q(s) g1, corresponding to a G1 point
    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment {
        let mut poly_to_commit = p - y;
        poly_to_commit.ruffini_division_inplace(x);
        self.commit(&poly_to_commit)
    }

    /// Verifies the correct evaluation of a polynomial p by providing a commitment to p,
    /// the point x, the evaluation y (p(x) = y) and an evaluation proof (commitment to the quotient polynomial)
    /// Basically, we want to show that, at secret point s, p(s) - y = (s - x) q(s)
    /// It uses pairings to verify the above condition, e(cm(p) - yg1,g2)*(cm(q), sg2 - xg2)^-1
    /// Returns true for valid evaluation
    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool {
        let g1 = &self.srs.powers_main_group[0];
        let g2 = &self.srs.powers_secondary_group[0];
        let alpha_g2 = &self.srs.powers_secondary_group[1];

        let e = P::compute_batch(&[
            (
                &p_commitment.operate_with(&(g1.operate_with_self(y.canonical())).neg()),
                g2,
            ),
            (
                &proof.neg(),
                &(alpha_g2.operate_with(&(g2.operate_with_self(x.canonical())).neg())),
            ),
        ]);
        e == Ok(FieldElement::one())
    }

    /// Creates an evaluation proof for several polynomials at a single point x. upsilon is used to
    /// perform the random linear combination, using Horner's evaluation form
    fn open_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        polynomials: &[Polynomial<FieldElement<F>>],
        upsilon: &FieldElement<F>,
    ) -> Self::Commitment {
        let acc_polynomial = polynomials
            .iter()
            .rev()
            .fold(Polynomial::zero(), |acc, polynomial| {
                acc * upsilon.to_owned() + polynomial
            });

        let acc_y = ys
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, y| acc * upsilon.to_owned() + y);

        self.open(x, &acc_y, &acc_polynomial)
    }

    /// Verifies an evaluation proof for the evaluation of a batch of polynomials at x, using upsilon to perform the random
    /// linear combination
    /// Outputs true if the evaluation is correct
    fn verify_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Commitment,
        upsilon: &FieldElement<F>,
    ) -> bool {
        let acc_commitment =
            p_commitments
                .iter()
                .rev()
                .fold(P::G1Point::neutral_element(), |acc, point| {
                    acc.operate_with_self(upsilon.to_owned().canonical())
                        .operate_with(point)
                });

        let acc_y = ys
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, y| acc * upsilon.to_owned() + y);
        self.verify(x, &acc_y, &acc_commitment, proof)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::slice;
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::{
                    curve::BLS12381Curve,
                    default_types::{FrElement, FrField},
                    pairing::BLS12381AtePairing,
                    twist::BLS12381TwistCurve,
                },
                point::ShortWeierstrassJacobianPoint,
            },
            traits::{IsEllipticCurve, IsPairing},
        },
        field::element::FieldElement,
        polynomial::Polynomial,
        traits::{AsBytes, Deserializable},
        unsigned_integer::element::U256,
    };

    use crate::commitments::traits::IsCommitmentScheme;

    use super::{KateZaveruchaGoldberg, StructuredReferenceString};
    use rand::Rng;

    type G1 = ShortWeierstrassJacobianPoint<BLS12381Curve>;

    #[allow(clippy::upper_case_acronyms)]
    type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

    fn create_srs() -> StructuredReferenceString<
        <BLS12381AtePairing as IsPairing>::G1Point,
        <BLS12381AtePairing as IsPairing>::G2Point,
    > {
        let mut rng = rand::thread_rng();
        let toxic_waste = FrElement::new(U256 {
            limbs: [
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
                rng.gen::<u64>(),
            ],
        });
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let powers_main_group: Vec<G1> = (0..100)
            .map(|exponent| g1.operate_with_self(toxic_waste.pow(exponent as u128).canonical()))
            .collect();
        let powers_secondary_group = [g2.clone(), g2.operate_with_self(toxic_waste.canonical())];
        StructuredReferenceString::new(&powers_main_group, &powers_secondary_group)
    }

    #[test]
    fn kzg_1() {
        let kzg = KZG::new(create_srs());
        let p = Polynomial::<FrElement>::new(&[FieldElement::one(), FieldElement::one()]);
        let p_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p);
        let x = -FieldElement::one();
        let y = p.evaluate(&x);
        let proof = kzg.open(&x, &y, &p);
        assert_eq!(y, FieldElement::zero());
        assert_eq!(proof, BLS12381Curve::generator());
        assert!(kzg.verify(&x, &y, &p_commitment, &proof));
    }

    #[test]
    fn poly_9000_constant_should_verify_proof() {
        let kzg = KZG::new(create_srs());
        let p = Polynomial::new(&[FieldElement::from(9000)]);
        let p_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p);
        let x = FieldElement::one();
        let y = FieldElement::from(9000);
        let proof = kzg.open(&x, &y, &p);
        assert!(kzg.verify(&x, &y, &p_commitment, &proof));
    }

    #[test]
    fn poly_9000_batched_should_verify() {
        let kzg = KZG::new(create_srs());
        let p0 = Polynomial::<FrElement>::new(&[FieldElement::from(9000)]);
        let p0_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p0);

        let x = FieldElement::one();
        let y0 = FieldElement::from(9000);
        let upsilon = &FieldElement::from(1);

        let proof = kzg.open_batch(&x, slice::from_ref(&y0), &[p0], upsilon);

        assert!(kzg.verify_batch(&x, &[y0], &[p0_commitment], &proof, upsilon));
    }

    #[test]
    fn two_poly_9000_batched_should_verify() {
        let kzg = KZG::new(create_srs());
        let p0 = Polynomial::<FrElement>::new(&[FieldElement::from(9000)]);
        let p0_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p0);

        let x = FieldElement::one();
        let y0 = FieldElement::from(9000);
        let upsilon = &FieldElement::from(1);

        let proof = kzg.open_batch(&x, &[y0.clone(), y0.clone()], &[p0.clone(), p0], upsilon);

        assert!(kzg.verify_batch(
            &x,
            &[y0.clone(), y0],
            &[p0_commitment.clone(), p0_commitment],
            &proof,
            upsilon
        ));
    }

    #[test]
    fn two_poly_batched_should_verify() {
        let kzg = KZG::new(create_srs());

        let x = FieldElement::from(3);

        let p0 = Polynomial::<FrElement>::new(&[FieldElement::from(9000)]); // Constant polynomial
        let p0_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p0);
        let y0 = FieldElement::from(9000);

        let p1 = Polynomial::<FrElement>::new(&[
            FieldElement::from(1),
            FieldElement::from(2),
            -FieldElement::from(1),
        ]); // p(x) = 1 + 2x - x²
        let p1_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p1);
        let y1 = p1.evaluate(&x);

        let upsilon = &FieldElement::from(1);

        let proof = kzg.open_batch(&x, &[y0.clone(), y1.clone()], &[p0, p1], upsilon);

        assert!(kzg.verify_batch(
            &x,
            &[y0, y1],
            &[p0_commitment, p1_commitment],
            &proof,
            upsilon
        ));
    }

    #[test]
    fn serialize_deserialize_srs() {
        let srs = create_srs();
        let bytes = srs.as_bytes();
        let deserialized: StructuredReferenceString<
            ShortWeierstrassJacobianPoint<BLS12381Curve>,
            ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
        > = StructuredReferenceString::deserialize(&bytes).unwrap();

        assert_eq!(srs, deserialized);
    }

    #[test]
    #[cfg(feature = "std")]
    fn save_and_load_srs_from_file() {
        type TestSrsType = StructuredReferenceString<
            ShortWeierstrassJacobianPoint<BLS12381Curve>,
            ShortWeierstrassJacobianPoint<BLS12381TwistCurve>,
        >;

        // Create a small SRS
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let powers_main_group: Vec<ShortWeierstrassJacobianPoint<BLS12381Curve>> =
            (0..3).map(|exp| g1.operate_with_self(exp as u64)).collect();
        let powers_secondary_group = [g2.clone(), g2.operate_with_self(2_u64)];
        let srs = TestSrsType::new(&powers_main_group, &powers_secondary_group);

        // Save to temp file
        let base_dir = env!("CARGO_MANIFEST_DIR");
        let srs_file = base_dir.to_owned() + "/src/commitments/test_srs/srs_3_g1_elements.bin";
        std::fs::write(&srs_file, srs.as_bytes()).unwrap();

        // Load back and verify
        let loaded_srs = TestSrsType::from_file(&srs_file).unwrap();
        assert_eq!(loaded_srs.powers_main_group.len(), 3);
        assert_eq!(srs, loaded_srs);
    }
}
