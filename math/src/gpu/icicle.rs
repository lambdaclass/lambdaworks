use icicle_bls12_377::{CurveCfg, G1Projective, G2CurveCfg, G2Projective, ScalarCfg};
use icicle_bls12_381::{CurveCfg, G1Projective, G2CurveCfg, G2Projective, ScalarCfg};
use icicle_bn254::{CurveCfg, G1Projective, G2CurveCfg, G2Projective, ScalarCfg};
use icicle_core::{
    field::Field,
    msm,
    traits::{FieldConfig, FieldImpl, GenerateRandom},
    Curve::{Affine, Curve, Projective},
    Field::{Field, FieldImpl, MontgomeryConvertibleField},
};
use icicle_cuda_runtime::{memory::HostOrDeviceSlice, stream::CudaStream};

use crate::{
    elliptic_curve::{
        short_weierstrass::{
            curves::{
                bls12_377::{
                    curve::{BLS12377Curve, BLS12377FieldElement},
                    field_extension::BLS12377PrimeField,
                },
                bls12_381::{
                    curve::{BLS12381Curve, BLS12381FieldElement, BLS12381TwistCurveFieldElement},
                    twist::BLS12381TwistCurve,
                },
                bn_254::{
                    curve::{BN254Curve, BN254FieldElement, BN254TwistCurveFieldElement},
                    twist::BN254TwistCurve,
                },
            },
            point::ShortWeierstrassProjectivePoint,
        },
        traits::IsEllipticCurve,
    },
    errors::ByteConversionError,
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
};

use core::fmt::Debug;

/// Notes:
/// Lambdaworks supplies rust bindings generic over there internal Field and Coordinate types.
/// The best solution is to upstream a `LambdaConvertible` trait implementation that handles this conversion for us.
/// In the meantime conversions are for specific curves and field implemented as the Icicle's Field type is not abstracted
/// from the field configuration or number of underlying limbs used in its representation

/// trait for Conversions of lambdaworks type -> Icicle type
/// NOTE: This may be removed with eliminating `LambdaConvertible`
pub trait ToIcicle: Clone + Debug {
    type IcicleType;

    fn to_icicle(&self) -> Self::IcicleType;
    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError>;
}

impl ToIcicle for BLS12377FieldElement {
    type IcicleType = icicle_bls12_377::curve::BaseField;

    fn to_icicle(&self) -> Self::IcicleType {
        IcicleType::from_bytes_le(self.to_representative().to_bytes_le())
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(icicle.to_repr().to_bytes_le())
    }
}

impl ToIcicle for BLS12381FieldElement {
    type IcicleType = icicle_bls12_381::curve::BaseField;

    fn to_icicle(&self) -> Self::IcicleType {
        IcicleType::from_bytes_le(self.to_representative().to_bytes_le())
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(icicle.to_repr().to_bytes_le())
    }
}

impl ToIcicle for BLS12381TwistCurveFieldElement {
    type IcicleType = icicle_bls12_381::curve::BaseField;

    fn to_icicle(&self) -> Self::IcicleType {
        IcicleType::from_bytes_le(self.to_representative().to_bytes_le())
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(icicle.to_repr().to_bytes_le())
    }
}

impl ToIcicle for BN254FieldElement {
    type IcicleType = icicle_bn254::curve::BaseField;

    fn to_icicle(&self) -> Self::IcicleType {
        IcicleType::from_bytes_le(self.to_representative().to_bytes_le())
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(icicle.to_repr().to_bytes_le())
    }
}

impl ToIcicle for BN254TwistCurveFieldElement {
    type IcicleType = icicle_bn254::curve::BaseField;

    fn to_icicle(&self) -> Self::IcicleType {
        IcicleType::from_bytes_le(self.to_representative().to_bytes_le())
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Self::from_bytes_le(&icicle.to_bytes_le())
    }
}

impl ToIcicle for ShortWeierstrassProjectivePoint<BLS12377Curve> {
    type IcicleType = icicle_bls12_377::curve::G1Projective;

    fn to_icicle(&self) -> Self::IcicleType {
        Self::IcicleType {
            x: self.x().to_icicle(),
            y: self.y().to_icicle(),
            z: self.z().to_icicle(),
        }
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            FieldElement::<BLS12377Curve>::from_icicle(icicle.x).unwrap(),
            FieldElement::<BLS12377Curve>::from_icicle(icicle.y).unwrap(),
            FieldElement::<BLS12377Curve>::from_icicle(icicle.z).unwrap(),
        ]))
    }
}

impl ToIcicle for ShortWeierstrassProjectivePoint<BLS12381Curve> {
    type IcicleType = icicle_bls12_3811::curve::G1Projective;

    fn to_icicle(&self) -> Self::IcicleType {
        Self::IcicleType {
            x: self.x().to_icicle(),
            y: self.y().to_icicle(),
            z: self.z().to_icicle(),
        }
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            FieldElement::<BLS12381Curve>::from_icicle(icicle.x).unwrap(),
            FieldElement::<BLS12381Curve>::from_icicle(icicle.y).unwrap(),
            FieldElement::<BLS12381Curve>::from_icicle(icicle.z).unwrap(),
        ]))
    }
}

impl ToIcicle for ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
    type IcicleType = icicle_bls12_381::curve::G2Projective;

    fn to_icicle(&self) -> Self::IcicleType {
        Self::IcicleType {
            x: self.x().to_icicle(),
            y: self.y().to_icicle(),
            z: self.z().to_icicle(),
        }
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            FieldElement::<BLS12381TwistCurve>::from_icicle(icicle.x).unwrap(),
            FieldElement::<BLS12381TwistCurve>::from_icicle(icicle.y).unwrap(),
            FieldElement::<BLS12381TwistCurve>::from_icicle(icicle.z).unwrap(),
        ]))
    }
}

impl ToIcicle for ShortWeierstrassProjectivePoint<BN254Curve> {
    type IcicleType = icicle_bn254::curve::G1Projective;

    fn to_icicle(&self) -> Self::IcicleType {
        Self::IcicleType {
            x: self.x().to_icicle(),
            y: self.y().to_icicle(),
            z: self.z().to_icicle(),
        }
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            FieldElement::<BN254Curve>::from_icicle(icicle.x).unwrap(),
            FieldElement::<BN254Curve>::from_icicle(icicle.y).unwrap(),
            FieldElement::<BN254Curve>::from_icicle(icicle.z).unwrap(),
        ]))
    }
}

impl ToIcicle for ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    type IcicleType = icicle_bn254::curve::G2Projective;

    fn to_icicle(&self) -> Self::IcicleType {
        Self::IcicleType {
            x: self.x().to_icicle(),
            y: self.y().to_icicle(),
            z: self.z().to_icicle(),
        }
    }

    fn from_icicle(icicle: Self::IcicleType) -> Result<Self, ByteConversionError> {
        Ok(Self::new([
            FieldElement::<BN254TwistCurve>::from_icicle(icicle.x).unwrap(),
            FieldElement::<BN254TwistCurve>::from_icicle(icicle.y).unwrap(),
            FieldElement::<BN254TwistCurve>::from_icicle(icicle.z).unwrap(),
        ]))
    }
}

/// Performs msm using Icicle GPU, intitiates, allocates, and configures all gpu operations
/// TODO: determining where this setup should occur is an open question
fn msm<F: IsField>(
    scalars: &[impl ToIcicle],
    points: &[impl ToIcicle],
) -> ShortWeierstrassProjectivePoint<F> {
    let scalars = HostOrDeviceSlice::Host(&scalars.iter().map(to_icicle()).collect::<Vec<_>>());
    let point = HostOrDeviceSlice::Host(&points.iter().map(to_icicle()).collect::<Vec<_>>());
    let mut msm_results = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = msm::get_default_msm_config();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
    let mut msm_host_result = Vec::new();
    stream.synchronize().unwrap();
    msm_results.copy_to_host(&mut msm_host_result[..]).unwrap();
    stream.destroy().unwrap();
}

/// Performs ntt using Icicle GPU, intitiates, allocates, and configures all gpu operations
fn ntt<F: IsField>(scalars: &[impl ToIcicle], points: &[impl ToIcicle]) -> FieldElement<F> {
    let point = HostOrDeviceSlice::Host(&points.iter().map(to_icicle()).collect::<Vec<_>>());
    let mut ntt_results = HostOrDeviceSlice::cuda_malloc(1).unwrap();
    let stream = CudaStream::create().unwrap();
    let mut cfg = msm::get_default_msm_config();
    cfg.ctx.stream = &stream;
    cfg.is_async = true;
    msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
    let mut ntt_host_result = Vec::new();
    stream.synchronize().unwrap();
    ntt_results.copy_to_host(&mut msm_host_result[..]).unwrap();
    stream.destroy().unwrap();

    let ctx = get_default_device_context();
}
