use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use crate::commitments::ipa::ipa::FrElement;
use crate::commitments::ipa::ipa::Proof;

pub trait IsCommitmentSchemeIPA {

    type CommitmentIPA;
    type OpenIPA;
    type VerifyOpen;

    fn commit<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve>(
        &self,
        p: &Polynomial<FrElement>, r: FrElement
    ) -> Self::CommitmentIPA;

    fn open(
        &mut self,
        a: &[FrElement],
        b: &[FrElement],
        u: &[FrElement],
        _u: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    ) -> Self::OpenIPA;

    fn verify_open(
        &self,
        x: &FrElement,
        v: &FrElement,
        _p: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
        p: &Proof<BLS12381Curve>,
        r: &FrElement,
        u: &[FrElement],
        _u: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    ) -> Self::VerifyOpen;

}
