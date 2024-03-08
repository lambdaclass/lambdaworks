//pub mod bls12_377;
//pub mod bls12_381;
//pub mod bn254;

use icicle_bls12_381::curve::CurveCfg as IcicleBLS12381Curve;
use icicle_bls12_377::curve::CurveCfg as IcicleBLS12377Curve;
use icicle_bn254::curve::CurveCfg as IcicleBN254Curve;
use icicle_cuda_runtime::{memory::HostOrDeviceSlice, stream::CudaStream};
use icicle_core::{error::IcicleError, msm, curve::{Curve, Affine, Projective}, traits::FieldImpl};
use crate::{
    elliptic_curve::{short_weierstrass::{
        curves::{
            bls12_381::curve::BLS12381Curve,
            bls12_377::curve::BLS12377Curve,
            bn_254::curve::BN254Curve
        },
        traits::IsShortWeierstrass, point::ShortWeierstrassProjectivePoint}, traits::IsEllipticCurve,
    },
    field::{element::FieldElement, traits::IsField},
    unsigned_integer::element::UnsignedInteger,
    cyclic_group::IsGroup,
    errors::ByteConversionError,
    traits::ByteConversion
};

use std::fmt::Debug;

impl Icicle<BLS12381Curve, IcicleBLS12381Curve> for BLS12381Curve {}
impl Icicle<BLS12377Curve, IcicleBLS12377Curve> for BLS12377Curve {}
impl Icicle<BN254Curve, IcicleBLS12381Curve> for BN254Curve {}

pub trait Icicle<C: IsShortWeierstrass + Clone + Debug, I: Curve>
where
    FieldElement<C::BaseField>: ByteConversion
{
    /// Used for searching this field's implementation in other languages, e.g in MSL
    /// for executing parallel operations with the Metal API.
    fn field_name() -> &'static str {
        ""
    }

    fn to_icicle_field(element: &FieldElement<C::BaseField>) -> I::BaseField {
        I::BaseField::from_bytes_le(&element.to_bytes_le())
    }

    fn to_icicle_scalar(element: &FieldElement<C::BaseField>) -> I::ScalarField {
        I::ScalarField::from_bytes_le(&element.to_bytes_le())
    }

    fn from_icicle_field(icicle: &I::BaseField) -> Result<FieldElement<C::BaseField>, ByteConversionError> {
        FieldElement::<C::BaseField>::from_bytes_le(&icicle.to_bytes_le())
    }

    fn to_icicle_affine(point: &ShortWeierstrassProjectivePoint<C>) -> Affine<I> {
        let s = ShortWeierstrassProjectivePoint::<C>::to_affine(point);
        Affine::<I> {
            x: Self::to_icicle_field(s.x()),
            y: Self::to_icicle_field(s.y()),
        }
    }

    fn from_icicle_projective(icicle: &Projective<I>) -> Result<ShortWeierstrassProjectivePoint<C>, ByteConversionError> {
        Ok(ShortWeierstrassProjectivePoint::<C>::new([
            Self::from_icicle_field(&icicle.x).unwrap(),
            Self::from_icicle_field(&icicle.y).unwrap(),
            Self::from_icicle_field(&icicle.z).unwrap(),
        ]))
    }

}

pub fn icicle_msm<C: IsShortWeierstrass + Clone + Debug, I: Curve + msm::MSM<I>>(
        scalars: &[FieldElement<C::BaseField>],
        points: &[ShortWeierstrassProjectivePoint<C>]
    ) -> Result<ShortWeierstrassProjectivePoint<C>, IcicleError> 
where 
    C: Icicle<C, I>,
    FieldElement<<C as IsEllipticCurve>::BaseField>: ByteConversion
{
    let mut cfg = msm::MSMConfig::default();
    let scalars = HostOrDeviceSlice::Host(
        scalars
            .iter()
            .map(|scalar| C::to_icicle_scalar(&scalar))
            .collect::<Vec<_>>(),
    );
    let points = HostOrDeviceSlice::Host(
        points
            .iter()
            .map(|point| C::to_icicle_affine(&point))
            .collect::<Vec<_>>(),
    );
    let mut msm_results = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
    let mut msm_host_result = vec![Projective::<I>::zero(); 1];
    stream.synchronize().unwrap();
    msm_results.copy_to_host(&mut msm_host_result[..]).unwrap();
    stream.destroy().unwrap();
    let res =
        C::from_icicle_projective(&msm_host_result[0]).unwrap();
    Ok(res)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::curve::BLS12381FieldElement,
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        msm::pippenger::msm,
    };

    impl ShortWeierstrassProjectivePoint<BLS12381Curve> {
        fn from_icicle_affine(
            icicle: &curve::G1Affine,
        ) -> Result<ShortWeierstrassProjectivePoint<BLS12381Curve>, ByteConversionError> {
            Ok(Self::new([
                FieldElement::<BLS12381PrimeField>::from_icicle(&icicle.x).unwrap(),
                FieldElement::<BLS12381PrimeField>::from_icicle(&icicle.y).unwrap(),
                FieldElement::one(),
            ]))
        }
    }

    fn point_times_5() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
        let x = BLS12381FieldElement::from_hex_unchecked(
            "32bcce7e71eb50384918e0c9809f73bde357027c6bf15092dd849aa0eac274d43af4c68a65fb2cda381734af5eecd5c",
        );
        let y = BLS12381FieldElement::from_hex_unchecked(
            "11e48467b19458aabe7c8a42dc4b67d7390fdf1e150534caadddc7e6f729d8890b68a5ea6885a21b555186452b954d88",
        );
        BLS12381Curve::create_point_from_affine(x, y).unwrap()
    }

    #[test]
    fn to_from_icicle() {
        // convert value of 5 to icicle and back again and that icicle 5 matches
        let point = point_times_5();
        let icicle_point = point.to_icicle();
        let res =
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_icicle_affine(&icicle_point)
                .unwrap();
        assert_eq!(point, res)
    }

    #[test]
    fn to_from_icicle_generator() {
        // Convert generator and see that it matches
        let point = BLS12381Curve::generator();
        let icicle_point = point.to_icicle();
        let res =
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_icicle_affine(&icicle_point)
                .unwrap();
        assert_eq!(point, res)
    }

    #[test]
    fn icicle_g1_msm() {
        const LEN: usize = 20;
        let eight: BLS12381FieldElement = FieldElement::from(8);
        let lambda_scalars = vec![eight; LEN];
        let lambda_points = (0..LEN).map(|_| point_times_5()).collect::<Vec<_>>();
        let expected = msm(
            &lambda_scalars,
            &lambda_points,
        )
        .unwrap();
        let res = bls12_381_g1_msm(&lambda_scalars, &lambda_points, None).unwrap();
        assert_eq!(res, expected);
    }
}
