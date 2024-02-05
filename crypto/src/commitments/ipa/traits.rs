use crate::commitments::ipa::pcs_ipa::Proof;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::polynomial::Polynomial;

pub trait IsCommitmentSchemeIPA<F: IsField + IsPrimeField, E: IsEllipticCurve + IsShortWeierstrass>
{
    type CommitmentIPA;
    type OpenIPA;
    type VerifyOpen;

    fn commit(
        &self,
        p: &Polynomial<FieldElement<F>>,
        zr: FieldElement<F>,
        neutral_element: ShortWeierstrassProjectivePoint<E>,
    ) -> Self::CommitmentIPA;

    fn open(
        &mut self,
        a: &[FieldElement<F>],
        b: &[FieldElement<F>],
        u: &[FieldElement<F>],
        u_point: &ShortWeierstrassProjectivePoint<E>,
        neutral_element: ShortWeierstrassProjectivePoint<E>,
    ) -> Self::OpenIPA;

    #[allow(clippy::too_many_arguments)]
    fn verify_open(
        &self,
        x: &FieldElement<F>,
        v: &FieldElement<F>,
        _p: &ShortWeierstrassProjectivePoint<E>,
        p: &Proof<F, E>,
        r: &FieldElement<F>,
        u: &[FieldElement<F>],
        u_point: &ShortWeierstrassProjectivePoint<E>,
        neutral_element: ShortWeierstrassProjectivePoint<E>,
    ) -> Self::VerifyOpen;
}
