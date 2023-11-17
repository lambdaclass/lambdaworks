
use super::{traits::IsCommitmentScheme, srs::StructuredReferenceString};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::IsPairing,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    unsigned_integer::element::UnsignedInteger, polynomial::Polynomial,
};
use std::marker::PhantomData;

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

impl<const N: usize, F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>, P: IsPairing>
    IsCommitmentScheme<F> for KateZaveruchaGoldberg<F, P>
{
    type Commitment = P::G1Point;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment {
        let coefficients: Vec<_> = p
            .coefficients
            .iter()
            .map(|coefficient| coefficient.representative())
            .collect();
        msm(
            &coefficients,
            &self.srs.powers_main_group[..coefficients.len()],
        )
        .expect("`points` is sliced by `cs`'s length")
    }

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
                &p_commitment.operate_with(&(g1.operate_with_self(y.representative())).neg()),
                g2,
            ),
            (
                &proof.neg(),
                &(alpha_g2.operate_with(&(g2.operate_with_self(x.representative())).neg())),
            ),
        ]);
        e == FieldElement::one()
    }

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
                    acc.operate_with_self(upsilon.to_owned().representative())
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
                point::ShortWeierstrassProjectivePoint,
            },
            traits::{IsEllipticCurve, IsPairing},
        },
        field::element::FieldElement,
        polynomial::Polynomial,
        traits::{Deserializable, Serializable},
        unsigned_integer::element::U256,
    };

    use crate::commitments::traits::IsCommitmentScheme;

    use super::{KateZaveruchaGoldberg, StructuredReferenceString};
    use rand::Rng;

    type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;

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
            .map(|exponent| {
                g1.operate_with_self(toxic_waste.pow(exponent as u128).representative())
            })
            .collect();
        let powers_secondary_group = [
            g2.clone(),
            g2.operate_with_self(toxic_waste.representative()),
        ];
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

        let proof = kzg.open_batch(&x, &[y0.clone()], &[p0], upsilon);

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

        let p0 = Polynomial::<FrElement>::new(&[FieldElement::from(9000)]);
        let p0_commitment: <BLS12381AtePairing as IsPairing>::G1Point = kzg.commit(&p0);
        let y0 = FieldElement::from(9000);

        let p1 = Polynomial::<FrElement>::new(&[
            FieldElement::from(1),
            FieldElement::from(2),
            -FieldElement::from(1),
        ]);
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
        let bytes = srs.serialize();
        let deserialized: StructuredReferenceString<
            ShortWeierstrassProjectivePoint<BLS12381Curve>,
            ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
        > = StructuredReferenceString::deserialize(&bytes).unwrap();

        assert_eq!(srs, deserialized);
    }

    #[test]
    fn load_srs_from_file() {
        type TestSrsType = StructuredReferenceString<
            ShortWeierstrassProjectivePoint<BLS12381Curve>,
            ShortWeierstrassProjectivePoint<BLS12381TwistCurve>,
        >;

        let base_dir = env!("CARGO_MANIFEST_DIR");
        let srs_file = base_dir.to_owned() + "/src/commitments/test_srs/srs_3_g1_elements.bin";

        let srs = TestSrsType::from_file(&srs_file).unwrap();

        assert_eq!(srs.powers_main_group.len(), 3);
    }
}
