//! Hash-to-curve for BLS12-381 G1 and G2 following RFC 9380.
//!
//! Implements the full pipeline:
//! 1. `hash_to_field` — expand message to field elements via expand_message_xmd (SHA-256)
//! 2. `map_to_curve` — Simplified SWU map to isogenous curve
//! 3. `iso_map` — Isogeny map (11-isogeny for G1, 3-isogeny for G2)
//! 4. `clear_cofactor` — Efficient cofactor clearing
//!
//! Suite identifiers:
//! - G1: `BLS12381G1_XMD:SHA-256_SSWU_RO_`
//! - G2: `BLS12381G2_XMD:SHA-256_SSWU_RO_`

use core::fmt;
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                curve::{BLS12381Curve, MILLER_LOOP_CONSTANT},
                field_extension::{BLS12381PrimeField, Degree2ExtensionField},
                twist::BLS12381TwistCurve,
            },
            point::ShortWeierstrassJacobianPoint,
        },
        traits::IsEllipticCurve,
    },
    field::{element::FieldElement, traits::IsField},
    traits::ByteConversion,
    unsigned_integer::element::U384,
};

use crate::hash::sha2::{Sha2Error, Sha2Hasher};

type FpElement = FieldElement<BLS12381PrimeField>;
type Fp2Element = FieldElement<Degree2ExtensionField>;
type G1Point = ShortWeierstrassJacobianPoint<BLS12381Curve>;
type G2Point = ShortWeierstrassJacobianPoint<BLS12381TwistCurve>;

/// Error type for hash-to-curve operations.
#[derive(Debug)]
pub enum HashToCurveError {
    /// Error from SHA-256 message expansion
    Sha2(Sha2Error),
}

impl fmt::Display for HashToCurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashToCurveError::Sha2(e) => write!(f, "hash-to-curve: {}", e),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HashToCurveError {}

impl From<Sha2Error> for HashToCurveError {
    fn from(e: Sha2Error) -> Self {
        HashToCurveError::Sha2(e)
    }
}

/// Default DST for BLS12-381 G1 hash-to-curve.
///
/// This value comes from the IETF BLS Signature draft (ciphersuite
/// `BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_`), not from RFC 9380 test vectors.
/// Callers implementing other protocols should supply their own DST.
pub const DEFAULT_DST_G1: &[u8] = b"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_";

/// Default DST for BLS12-381 G2 hash-to-curve.
///
/// This value comes from the IETF BLS Signature draft (ciphersuite
/// `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_`), not from RFC 9380 test vectors.
/// Callers implementing other protocols should supply their own DST.
pub const DEFAULT_DST_G2: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

// 2^384 mod p — used for efficient os2ip_mod_p reduction.
const R_384: FpElement = FpElement::from_hex_unchecked(
    "15f65ec3fa80e4935c071a97a256ec6d77ce5853705257455f48985753c758baebf4000bc40c0002760900000002fffd",
);

// ============================================================================
// G1 constants
// ============================================================================

// SSWU parameters for E1': y² = x³ + A'x + B', Z = 11
const SSWU_A_G1: FpElement = FpElement::from_hex_unchecked(
    "144698a3b8e9433d693a02c96d4982b0ea985383ee66a8d8e8981aefd881ac98936f8da0e0f97f5cf428082d584c1d",
);
const SSWU_B_G1: FpElement = FpElement::from_hex_unchecked(
    "12e2908d11688030018b12e8753eee3b2016c1f0f24f4070a0b9c14fcef35ef55a23215a316ceaa5d1cc48e98e172be0",
);
const SSWU_Z_G1: FpElement = FpElement::from_hex_unchecked("b"); // 11

// ============================================================================
// G2 constants
// ============================================================================

// SSWU parameters for E2': y² = x³ + 240i·x + 1012(1+i), Z = -(2+i)
const SSWU_A_G2: Fp2Element = Fp2Element::const_from_raw([
    FpElement::from_hex_unchecked("0"),
    FpElement::from_hex_unchecked("f0"), // 240
]);
const SSWU_B_G2: Fp2Element = Fp2Element::const_from_raw([
    FpElement::from_hex_unchecked("3f4"), // 1012
    FpElement::from_hex_unchecked("3f4"), // 1012
]);
// Z = -(2+i) = (-2, -1) in Fp2
const SSWU_Z_G2: Fp2Element = Fp2Element::const_from_raw([
    // -2 mod p
    FpElement::from_hex_unchecked(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaa9",
    ),
    // -1 mod p
    FpElement::from_hex_unchecked(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaaa",
    ),
]);

// ============================================================================
// 11-isogeny coefficients for G1 (RFC 9380 Section 8.8.1)
// ============================================================================

const ISO_G1_XNUM: [FpElement; 12] = [
    FpElement::from_hex_unchecked("11a05f2b1e833340b809101dd99815856b303e88a2d7005ff2627b56cdb4e2c85610c2d5f2e62d6eaeac1662734649b7"),
    FpElement::from_hex_unchecked("17294ed3e943ab2f0588bab22147a81c7c17e75b2f6a8417f565e33c70d1e86b4838f2a6f318c356e834eef1b3cb83bb"),
    FpElement::from_hex_unchecked("0d54005db97678ec1d1048c5d10a9a1bce032473295983e56878e501ec68e25c958c3e3d2a09729fe0179f9dac9edcb0"),
    FpElement::from_hex_unchecked("1778e7166fcc6db74e0609d307e55412d7f5e4656a8dbf25f1b33289f1b330835336e25ce3107193c5b388641d9b6861"),
    FpElement::from_hex_unchecked("0e99726a3199f4436642b4b3e4118e5499db995a1257fb3f086eeb65982fac18985a286f301e77c451154ce9ac8895d9"),
    FpElement::from_hex_unchecked("1630c3250d7313ff01d1201bf7a74ab5db3cb17dd952799b9ed3ab9097e68f90a0870d2dcae73d19cd13c1c66f652983"),
    FpElement::from_hex_unchecked("0d6ed6553fe44d296a3726c38ae652bfb11586264f0f8ce19008e218f9c86b2a8da25128c1052ecaddd7f225a139ed84"),
    FpElement::from_hex_unchecked("17b81e7701abdbe2e8743884d1117e53356de5ab275b4db1a682c62ef0f2753339b7c8f8c8f475af9ccb5618e3f0c88e"),
    FpElement::from_hex_unchecked("080d3cf1f9a78fc47b90b33563be990dc43b756ce79f5574a2c596c928c5d1de4fa295f296b74e956d71986a8497e317"),
    FpElement::from_hex_unchecked("169b1f8e1bcfa7c42e0c37515d138f22dd2ecb803a0c5c99676314baf4bb1b7fa3190b2edc0327797f241067be390c9e"),
    FpElement::from_hex_unchecked("10321da079ce07e272d8ec09d2565b0dfa7dccdde6787f96d50af36003b14866f69b771f8c285decca67df3f1605fb7b"),
    FpElement::from_hex_unchecked("06e08c248e260e70bd1e962381edee3d31d79d7e22c837bc23c0bf1bc24c6b68c24b1b80b64d391fa9c8ba2e8ba2d229"),
];

const ISO_G1_XDEN: [FpElement; 11] = [
    FpElement::from_hex_unchecked("08ca8d548cff19ae18b2e62f4bd3fa6f01d5ef4ba35b48ba9c9588617fc8ac62b558d681be343df8993cf9fa40d21b1c"),
    FpElement::from_hex_unchecked("12561a5deb559c4348b4711298e536367041e8ca0cf0800c0126c2588c48bf5713daa8846cb026e9e5c8276ec82b3bff"),
    FpElement::from_hex_unchecked("0b2962fe57a3225e8137e629bff2991f6f89416f5a718cd1fca64e00b11aceacd6a3d0967c94fedcfcc239ba5cb83e19"),
    FpElement::from_hex_unchecked("03425581a58ae2fec83aafef7c40eb545b08243f16b1655154cca8abc28d6fd04976d5243eecf5c4130de8938dc62cd8"),
    FpElement::from_hex_unchecked("13a8e162022914a80a6f1d5f43e7a07dffdfc759a12062bb8d6b44e833b306da9bd29ba81f35781d539d395b3532a21e"),
    FpElement::from_hex_unchecked("0e7355f8e4e667b955390f7f0506c6e9395735e9ce9cad4d0a43bcef24b8982f7400d24bc4228f11c02df9a29f6304a5"),
    FpElement::from_hex_unchecked("0772caacf16936190f3e0c63e0596721570f5799af53a1894e2e073062aede9cea73b3538f0de06cec2574496ee84a3a"),
    FpElement::from_hex_unchecked("14a7ac2a9d64a8b230b3f5b074cf01996e7f63c21bca68a81996e1cdf9822c580fa5b9489d11e2d311f7d99bbdcc5a5e"),
    FpElement::from_hex_unchecked("0a10ecf6ada54f825e920b3dafc7a3cce07f8d1d7161366b74100da67f39883503826692abba43704776ec3a79a1d641"),
    FpElement::from_hex_unchecked("095fc13ab9e92ad4476d6e3eb3a56680f682b4ee96f7d03776df533978f31c1593174e4b4b7865002d6384d168ecdd0a"),
    FpElement::from_hex_unchecked("1"), // monic
];

const ISO_G1_YNUM: [FpElement; 16] = [
    FpElement::from_hex_unchecked("090d97c81ba24ee0259d1f094980dcfa11ad138e48a869522b52af6c956543d3cd0c7aee9b3ba3c2be9845719707bb33"),
    FpElement::from_hex_unchecked("134996a104ee5811d51036d776fb46831223e96c254f383d0f906343eb67ad34d6c56711962fa8bfe097e75a2e41c696"),
    FpElement::from_hex_unchecked("00cc786baa966e66f4a384c86a3b49942552e2d658a31ce2c344be4b91400da7d26d521628b00523b8dfe240c72de1f6"),
    FpElement::from_hex_unchecked("01f86376e8981c217898751ad8746757d42aa7b90eeb791c09e4a3ec03251cf9de405aba9ec61deca6355c77b0e5f4cb"),
    FpElement::from_hex_unchecked("08cc03fdefe0ff135caf4fe2a21529c4195536fbe3ce50b879833fd221351adc2ee7f8dc099040a841b6daecf2e8fedb"),
    FpElement::from_hex_unchecked("16603fca40634b6a2211e11db8f0a6a074a7d0d4afadb7bd76505c3d3ad5544e203f6326c95a807299b23ab13633a5f0"),
    FpElement::from_hex_unchecked("04ab0b9bcfac1bbcb2c977d027796b3ce75bb8ca2be184cb5231413c4d634f3747a87ac2460f415ec961f8855fe9d6f2"),
    FpElement::from_hex_unchecked("0987c8d5333ab86fde9926bd2ca6c674170a05bfe3bdd81ffd038da6c26c842642f64550fedfe935a15e4ca31870fb29"),
    FpElement::from_hex_unchecked("09fc4018bd96684be88c9e221e4da1bb8f3abd16679dc26c1e8b6e6a1f20cabe69d65201c78607a360370e577bdba587"),
    FpElement::from_hex_unchecked("0e1bba7a1186bdb5223abde7ada14a23c42a0ca7915af6fe06985e7ed1e4d43b9b3f7055dd4eba6f2bafaaebca731c30"),
    FpElement::from_hex_unchecked("19713e47937cd1be0dfd0b8f1d43fb93cd2fcbcb6caf493fd1183e416389e61031bf3a5cce3fbafce813711ad011c132"),
    FpElement::from_hex_unchecked("18b46a908f36f6deb918c143fed2edcc523559b8aaf0c2462e6bfe7f911f643249d9cdf41b44d606ce07c8a4d0074d8e"),
    FpElement::from_hex_unchecked("0b182cac101b9399d155096004f53f447aa7b12a3426b08ec02710e807b4633f06c851c1919211f20d4c04f00b971ef8"),
    FpElement::from_hex_unchecked("0245a394ad1eca9b72fc00ae7be315dc757b3b080d4c158013e6632d3c40659cc6cf90ad1c232a6442d9d3f5db980133"),
    FpElement::from_hex_unchecked("05c129645e44cf1102a159f748c4a3fc5e673d81d7e86568d9ab0f5d396a7ce46ba1049b6579afb7866b1e715475224b"),
    FpElement::from_hex_unchecked("15e6be4e990f03ce4ea50b3b42df2eb5cb181d8f84965a3957add4fa95af01b2b665027efec01c7704b456be69c8b604"),
];

const ISO_G1_YDEN: [FpElement; 16] = [
    FpElement::from_hex_unchecked("16112c4c3a9c98b252181140fad0eae9601a6de578980be6eec3232b5be72e7a07f3688ef60c206d01479253b03663c1"),
    FpElement::from_hex_unchecked("1962d75c2381201e1a0cbd6c43c348b885c84ff731c4d59ca4a10356f453e01f78a4260763529e3532f6102c2e49a03d"),
    FpElement::from_hex_unchecked("058df3306640da276faaae7d6e8eb15778c4855551ae7f310c35a5dd279cd2eca6757cd636f96f891e2538b53dbf67f2"),
    FpElement::from_hex_unchecked("16b7d288798e5395f20d23bf89edb4d1d115c5dbddbcd30e123da489e726af41727364f2c28297ada8d26d98445f5416"),
    FpElement::from_hex_unchecked("0be0e079545f43e4b00cc912f8228ddcc6d19c9f0f69bbb0542eda0fc9dec916a20b15dc0fd2ededda39142311a5001d"),
    FpElement::from_hex_unchecked("08d9e5297186db2d9fb266eaac783182b70152c65550d881c5ecd87b6f0f5a6449f38db9dfa9cce202c6477faaf9b7ac"),
    FpElement::from_hex_unchecked("166007c08a99db2fc3ba8734ace9824b5eecfdfa8d0cf8ef5dd365bc400a0051d5fa9c01a58b1fb93d1a1399126a775c"),
    FpElement::from_hex_unchecked("16a3ef08be3ea7ea03bcddfabba6ff6ee5a4375efa1f4fd7feb34fd206357132b920f5b00801dee460ee415a15812ed9"),
    FpElement::from_hex_unchecked("1866c8ed336c61231a1be54fd1d74cc4f9fb0ce4c6af5920abc5750c4bf39b4852cfe2f7bb9248836b233d9d55535d4a"),
    FpElement::from_hex_unchecked("167a55cda70a6e1cea820597d94a84903216f763e13d87bb5308592e7ea7d4fbc7385ea3d529b35e346ef48bb8913f55"),
    FpElement::from_hex_unchecked("04d2f259eea405bd48f010a01ad2911d9c6dd039bb61a6290e591b36e636a5c871a5c29f4f83060400f8b49cba8f6aa8"),
    FpElement::from_hex_unchecked("0accbb67481d033ff5852c1e48c50c477f94ff8aefce42d28c0f9a88cea7913516f968986f7ebbea9684b529e2561092"),
    FpElement::from_hex_unchecked("0ad6b9514c767fe3c3613144b45f1496543346d98adf02267d5ceef9a00d9b8693000763e3b90ac11e99b138573345cc"),
    FpElement::from_hex_unchecked("02660400eb2e4f3b628bdd0d53cd76f2bf565b94e72927c1cb748df27942480e420517bd8714cc80d1fadc1326ed06f7"),
    FpElement::from_hex_unchecked("0e0fa1d816ddc03e6b24255e0d7819c171c40f65e273b853324efcd6356caa205ca2f570f13497804415473a1d634b8f"),
    FpElement::from_hex_unchecked("1"), // monic
];

// ψ² endomorphism x-coordinate coefficient.
const PSI2_COEFF_X: Fp2Element = Fp2Element::const_from_raw([
    FpElement::from_hex_unchecked(
        "1a0111ea397fe699ec02408663d4de85aa0d857d89759ad4897d29650fb85f9b409427eb4f49fffd8bfd00000000aaac",
    ),
    FpElement::from_hex_unchecked("0"),
]);

// ============================================================================
// 3-isogeny coefficients for G2 (RFC 9380 Section 8.8.2)
// ============================================================================

const fn fp2_const(c0: &str, c1: &str) -> Fp2Element {
    Fp2Element::const_from_raw([
        FpElement::from_hex_unchecked(c0),
        FpElement::from_hex_unchecked(c1),
    ])
}

const ISO_G2_XNUM: [Fp2Element; 4] = [
    fp2_const(
        "05c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97d6",
        "05c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97d6",
    ),
    fp2_const(
        "0",
        "11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71a",
    ),
    fp2_const(
        "11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71e",
        "08ab05f8bdd54cde190937e76bc3e447cc27c3d6fbd7063fcd104635a790520c0a395554e5c6aaaa9354ffffffffe38d",
    ),
    fp2_const(
        "171d6541fa38ccfaed6dea691f5fb614cb14b4e7f4e810aa22d6108f142b85757098e38d0f671c7188e2aaaaaaaa5ed1",
        "0",
    ),
];

const ISO_G2_XDEN: [Fp2Element; 3] = [
    fp2_const(
        "0",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa63",
    ),
    fp2_const(
        "0c",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa9f",
    ),
    fp2_const("1", "0"), // monic
];

const ISO_G2_YNUM: [Fp2Element; 4] = [
    fp2_const(
        "1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf54439d87d27e500fc8c25ebf8c92f6812cfc71c71c6d706",
        "1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf54439d87d27e500fc8c25ebf8c92f6812cfc71c71c6d706",
    ),
    fp2_const(
        "0",
        "05c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97be",
    ),
    fp2_const(
        "11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71c",
        "08ab05f8bdd54cde190937e76bc3e447cc27c3d6fbd7063fcd104635a790520c0a395554e5c6aaaa9354ffffffffe38f",
    ),
    fp2_const(
        "124c9ad43b6cf79bfbf7043de3811ad0761b0f37a1e26286b0e977c69aa274524e79097a56dc4bd9e1b371c71c718b10",
        "0",
    ),
];

const ISO_G2_YDEN: [Fp2Element; 4] = [
    fp2_const(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb",
    ),
    fp2_const(
        "0",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa9d3",
    ),
    fp2_const(
        "12",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa99",
    ),
    fp2_const("1", "0"), // monic
];

// ============================================================================
// Public API
// ============================================================================

/// Hash a message to a point on BLS12-381 G1 (RFC 9380).
///
/// Uses the full hash-to-curve pipeline: hash_to_field → SSWU → 11-isogeny → cofactor clearing.
/// The result is always a valid point in the prime-order G1 subgroup.
pub fn hash_to_g1(msg: &[u8], dst: &[u8]) -> Result<G1Point, HashToCurveError> {
    let (u0, u1) = hash_to_fp(msg, dst)?;
    let (q0x, q0y) = map_to_curve_sswu_g1(&u0);
    let (q1x, q1y) = map_to_curve_sswu_g1(&u1);
    let (r0x, r0y) = iso_map_g1(&q0x, &q0y);
    let (r1x, r1y) = iso_map_g1(&q1x, &q1y);
    let r0 = BLS12381Curve::create_point_from_affine(r0x, r0y)
        .expect("iso_map output should be on curve");
    let r1 = BLS12381Curve::create_point_from_affine(r1x, r1y)
        .expect("iso_map output should be on curve");
    Ok(clear_cofactor_g1(&r0.operate_with(&r1)))
}

/// Hash a message to G1 with the default DST.
pub fn hash_to_g1_default(msg: &[u8]) -> Result<G1Point, HashToCurveError> {
    hash_to_g1(msg, DEFAULT_DST_G1)
}

/// Hash a message to a point on BLS12-381 G2 (RFC 9380).
///
/// Uses the full hash-to-curve pipeline: hash_to_field → SSWU → 3-isogeny → cofactor clearing.
/// The result is always a valid point in the prime-order G2 subgroup.
pub fn hash_to_g2(msg: &[u8], dst: &[u8]) -> Result<G2Point, HashToCurveError> {
    let (u0, u1) = hash_to_fp2(msg, dst)?;
    let (q0x, q0y) = map_to_curve_sswu_g2(&u0);
    let (q1x, q1y) = map_to_curve_sswu_g2(&u1);
    let (r0x, r0y) = iso_map_g2(&q0x, &q0y);
    let (r1x, r1y) = iso_map_g2(&q1x, &q1y);
    let r0 = BLS12381TwistCurve::create_point_from_affine(r0x, r0y)
        .expect("iso_map output should be on curve");
    let r1 = BLS12381TwistCurve::create_point_from_affine(r1x, r1y)
        .expect("iso_map output should be on curve");
    Ok(clear_cofactor_g2(&r0.operate_with(&r1)))
}

/// Hash a message to G2 with the default DST.
pub fn hash_to_g2_default(msg: &[u8]) -> Result<G2Point, HashToCurveError> {
    hash_to_g2(msg, DEFAULT_DST_G2)
}

// ============================================================================
// hash_to_field
// ============================================================================

/// Expand message and produce two Fp elements for G1.
fn hash_to_fp(msg: &[u8], dst: &[u8]) -> Result<(FpElement, FpElement), HashToCurveError> {
    const L: usize = 64;
    let expanded = Sha2Hasher::expand_message(msg, dst, (2 * L) as u64)?;
    Ok((
        os2ip_mod_p(&expanded[0..L]),
        os2ip_mod_p(&expanded[L..2 * L]),
    ))
}

/// Expand message and produce two Fp2 elements for G2.
fn hash_to_fp2(msg: &[u8], dst: &[u8]) -> Result<(Fp2Element, Fp2Element), HashToCurveError> {
    const L: usize = 64;
    let expanded = Sha2Hasher::expand_message(msg, dst, (4 * L) as u64)?;
    let e0 = os2ip_mod_p(&expanded[0..L]);
    let e1 = os2ip_mod_p(&expanded[L..2 * L]);
    let e2 = os2ip_mod_p(&expanded[2 * L..3 * L]);
    let e3 = os2ip_mod_p(&expanded[3 * L..4 * L]);
    Ok((Fp2Element::new([e0, e1]), Fp2Element::new([e2, e3])))
}

/// Convert 64 bytes (big-endian) to a field element mod p.
///
/// Splits the 512-bit integer into high (128 bits) and low (384 bits),
/// then computes: high * 2^384 + low (mod p). Cost: 1 field mul + 1 field add.
fn os2ip_mod_p(bytes: &[u8]) -> FpElement {
    debug_assert_eq!(bytes.len(), 64);
    // High 16 bytes → 128-bit value in a U384
    let mut high_bytes = [0u8; 48];
    high_bytes[48 - 16..].copy_from_slice(&bytes[0..16]);
    let high = FpElement::new(U384::from_bytes_be(&high_bytes).unwrap());

    // Low 48 bytes → 384-bit value
    let low = FpElement::new(U384::from_bytes_be(&bytes[16..64]).unwrap());

    high * R_384 + low
}

// ============================================================================
// SSWU map
// ============================================================================

/// SSWU map for G1: Fp → point on E1' (isogenous curve).
fn map_to_curve_sswu_g1(u: &FpElement) -> (FpElement, FpElement) {
    let a = &SSWU_A_G1;
    let b = &SSWU_B_G1;
    let z = &SSWU_Z_G1;

    let u2 = u * u;
    let u4 = &u2 * &u2;
    let z2 = z * z;
    let tv1_denom = &z2 * &u4 + z * &u2;

    let tv1 = if tv1_denom == FpElement::zero() {
        FpElement::zero()
    } else {
        tv1_denom.inv().expect("nonzero after check")
    };

    let neg_b_over_a = -b * a.inv().expect("A' is nonzero");
    let x1 = if tv1 == FpElement::zero() {
        b * (z * a).inv().expect("Z*A is nonzero")
    } else {
        &neg_b_over_a * (FpElement::one() + &tv1)
    };

    let gx1 = &x1 * &x1 * &x1 + a * &x1 + b;
    let x2 = z * &u2 * &x1;
    let gx2 = &x2 * &x2 * &x2 + a * &x2 + b;

    let (x, y) = if let Some(y1) = fp_sqrt(&gx1) {
        (x1, y1)
    } else if let Some(y2) = fp_sqrt(&gx2) {
        (x2, y2)
    } else {
        panic!("SSWU G1: neither gx1 nor gx2 is a square");
    };

    let y = if sgn0_fp(u) != sgn0_fp(&y) { -y } else { y };
    (x, y)
}

/// SSWU map for G2: Fp2 → point on E2' (isogenous curve).
fn map_to_curve_sswu_g2(u: &Fp2Element) -> (Fp2Element, Fp2Element) {
    let a = &SSWU_A_G2;
    let b = &SSWU_B_G2;
    let z = &SSWU_Z_G2;

    let u2 = u * u;
    let u4 = &u2 * &u2;
    let z2 = z * z;
    let tv1_denom = &z2 * &u4 + z * &u2;

    let tv1 = if tv1_denom == Fp2Element::zero() {
        Fp2Element::zero()
    } else {
        tv1_denom.inv().expect("nonzero after check")
    };

    let neg_b_over_a = -b * a.inv().expect("A' is nonzero");
    let x1 = if tv1 == Fp2Element::zero() {
        b * (z * a).inv().expect("Z*A is nonzero")
    } else {
        &neg_b_over_a * (Fp2Element::one() + &tv1)
    };

    let gx1 = &x1 * &x1 * &x1 + a * &x1 + b;
    let x2 = z * &u2 * &x1;
    let gx2 = &x2 * &x2 * &x2 + a * &x2 + b;

    let (x, y) = if let Some(y1) = fp2_sqrt(&gx1) {
        (x1, y1)
    } else if let Some(y2) = fp2_sqrt(&gx2) {
        (x2, y2)
    } else {
        panic!("SSWU G2: neither gx1 nor gx2 is a square");
    };

    let y = if sgn0_fp2(u) != sgn0_fp2(&y) { -y } else { y };
    (x, y)
}

// ============================================================================
// Isogeny maps
// ============================================================================

/// Evaluate a polynomial at x using Horner's method.
/// Coefficients are [c0, c1, ..., cn] representing c0 + c1*x + ... + cn*x^n.
fn eval_poly<F: IsField>(x: &FieldElement<F>, coeffs: &[FieldElement<F>]) -> FieldElement<F> {
    let mut result = coeffs.last().expect("coeffs must be non-empty").clone();
    for c in coeffs.iter().rev().skip(1) {
        result = &result * x + c;
    }
    result
}

/// 11-isogeny from E1' to E1 (RFC 9380 Section 8.8.1).
fn iso_map_g1(x: &FpElement, y: &FpElement) -> (FpElement, FpElement) {
    let x_num = eval_poly(x, &ISO_G1_XNUM);
    let x_den = eval_poly(x, &ISO_G1_XDEN);
    let y_num = eval_poly(x, &ISO_G1_YNUM);
    let y_den = eval_poly(x, &ISO_G1_YDEN);
    let x_out = &x_num * x_den.inv().expect("x_den nonzero");
    let y_out = y * &y_num * y_den.inv().expect("y_den nonzero");
    (x_out, y_out)
}

/// 3-isogeny from E2' to E2 (RFC 9380 Section 8.8.2).
fn iso_map_g2(x: &Fp2Element, y: &Fp2Element) -> (Fp2Element, Fp2Element) {
    let x_num = eval_poly(x, &ISO_G2_XNUM);
    let x_den = eval_poly(x, &ISO_G2_XDEN);
    let y_num = eval_poly(x, &ISO_G2_YNUM);
    let y_den = eval_poly(x, &ISO_G2_YDEN);
    let x_out = &x_num * x_den.inv().expect("x_den nonzero");
    let y_out = y * &y_num * y_den.inv().expect("y_den nonzero");
    (x_out, y_out)
}

// ============================================================================
// Cofactor clearing
// ============================================================================

/// Cofactor clearing for G1: h_eff(P) = [1 - x]P = [1 + |BLS_X|]P.
fn clear_cofactor_g1(p: &G1Point) -> G1Point {
    p.operate_with_self(MILLER_LOOP_CONSTANT).operate_with(p)
}

/// Efficient cofactor clearing for G2 using endomorphisms (Budroni-Pintore method).
///
/// Computes h_eff(P) = [x²-x-1]P + [x-1]·ψ(P) + ψ²(2P)
fn clear_cofactor_g2(p: &G2Point) -> G2Point {
    if p.is_neutral_element() {
        return p.clone();
    }
    let x = MILLER_LOOP_CONSTANT;
    let x_p = p.operate_with_self(x).neg(); // [x]P (x is negative)
    let psi_p = p.psi();
    let psi2_2p = psi_square(&p.double());

    let t3 = psi2_2p.operate_with(&psi_p.neg());
    let t2 = x_p.operate_with(&psi_p);
    let t2 = t2.operate_with_self(x).neg();
    let t3 = t3.operate_with(&t2);
    let t3 = t3.operate_with(&x_p.neg());
    t3.operate_with(&p.neg())
}

/// ψ² endomorphism on G2.
fn psi_square(p: &G2Point) -> G2Point {
    if p.is_neutral_element() {
        return p.clone();
    }
    let [x, y, z] = p.coordinates();
    ShortWeierstrassJacobianPoint::new([x * &PSI2_COEFF_X, -y, z.clone()])
        .expect("psi_square preserves curve validity")
}

// ============================================================================
// Field utilities
// ============================================================================

/// sgn0 for Fp (RFC 9380): returns true if x is odd.
fn sgn0_fp(x: &FpElement) -> bool {
    let bytes = x.canonical().to_bytes_be();
    bytes.last().map(|b| b & 1 == 1).unwrap_or(false)
}

/// sgn0 for Fp2 (RFC 9380).
fn sgn0_fp2(x: &Fp2Element) -> bool {
    let [x0, x1] = x.value();
    let sign_0 = sgn0_fp(x0);
    let zero_0 = *x0 == FpElement::zero();
    let sign_1 = sgn0_fp(x1);
    sign_0 || (zero_0 && sign_1)
}

/// Square root in Fp. For BLS12-381, p ≡ 3 (mod 4), so sqrt(a) = a^((p+1)/4).
fn fp_sqrt(a: &FpElement) -> Option<FpElement> {
    if *a == FpElement::zero() {
        return Some(FpElement::zero());
    }
    // (p + 1) / 4
    let exp = U384::from_hex_unchecked(
        "0680447a8e5ff9a692c6e9ed90d2eb35d91dd2e13ce144afd9cc34a83dac3d8907aaffffac54ffffee7fbfffffffeaab",
    );
    let sqrt = a.pow(exp);
    if &sqrt * &sqrt == *a {
        Some(sqrt)
    } else {
        None
    }
}

/// Square root in Fp2.
///
/// For BLS12-381 (p ≡ 3 mod 4), when a1 ≠ 0 exactly one of gamma/gamma2 is a
/// quadratic residue, its square root delta is non-zero (delta = 0 would imply
/// a1 = 0, a contradiction), and the constructed result satisfies result² = a
/// by construction. The debug_asserts below document these algebraic invariants;
/// the fallible guards are kept as defense-in-depth in release mode.
fn fp2_sqrt(a: &Fp2Element) -> Option<Fp2Element> {
    let [a0, a1] = a.value();

    if *a1 == FpElement::zero() {
        if let Some(s) = fp_sqrt(a0) {
            return Some(Fp2Element::new([s, FpElement::zero()]));
        }
        let neg_a0 = -a0;
        if let Some(s) = fp_sqrt(&neg_a0) {
            return Some(Fp2Element::new([FpElement::zero(), s]));
        }
        return None;
    }

    let alpha = &a0.square() + &a1.square();
    let sqrt_alpha = fp_sqrt(&alpha)?;
    let two_inv = FpElement::from(2u64).inv().ok()?;

    let gamma = (a0 + &sqrt_alpha) * &two_inv;
    if let Some(delta) = fp_sqrt(&gamma) {
        let two_delta = &delta + &delta;
        // delta = 0 implies gamma = 0 implies a0 = -sqrt_alpha implies a1² = 0,
        // which contradicts the a1 ≠ 0 guard above.
        debug_assert!(
            two_delta != FpElement::zero(),
            "fp2_sqrt: delta cannot be zero when a1 != 0"
        );
        if two_delta != FpElement::zero() {
            let y1 = a1 * two_delta.inv().ok()?;
            let result = Fp2Element::new([delta.clone(), y1]);
            debug_assert!(
                &result * &result == *a,
                "fp2_sqrt: result² != a (should be unreachable)"
            );
            if &result * &result == *a {
                return Some(result);
            }
        }
    }

    let gamma2 = (a0 - &sqrt_alpha) * &two_inv;
    if let Some(delta2) = fp_sqrt(&gamma2) {
        let two_delta2 = &delta2 + &delta2;
        debug_assert!(
            two_delta2 != FpElement::zero(),
            "fp2_sqrt: delta2 cannot be zero when a1 != 0"
        );
        if two_delta2 != FpElement::zero() {
            let y1 = a1 * two_delta2.inv().ok()?;
            let result = Fp2Element::new([delta2, y1]);
            debug_assert!(
                &result * &result == *a,
                "fp2_sqrt: result² != a (should be unreachable)"
            );
            if &result * &result == *a {
                return Some(result);
            }
        }
    }

    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;

    // RFC 9380 Appendix J.9/J.10 test DSTs
    const RFC_DST_G1: &[u8] = b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_";
    const RFC_DST_G2: &[u8] = b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_RO_";

    #[test]
    fn test_rfc9380_g1_empty_message() {
        let p = hash_to_g1(b"", RFC_DST_G1).unwrap().to_affine();
        let [px, py, _] = p.coordinates();

        let expected_x = FpElement::new(U384::from_hex_unchecked(
            "052926add2207b76ca4fa57a8734416c8dc95e24501772c814278700eed6d1e4e8cf62d9c09db0fac349612b759e79a1",
        ));
        let expected_y = FpElement::new(U384::from_hex_unchecked(
            "08ba738453bfed09cb546dbb0783dbb3a5f1f566ed67bb6be0e8c67e2e81a4cc68ee29813bb7994998f3eae0c9c6a265",
        ));

        assert_eq!(*px, expected_x, "G1 empty msg: x mismatch");
        assert_eq!(*py, expected_y, "G1 empty msg: y mismatch");
    }

    #[test]
    fn test_rfc9380_g1_abc() {
        let p = hash_to_g1(b"abc", RFC_DST_G1).unwrap().to_affine();
        let [px, py, _] = p.coordinates();

        let expected_x = FpElement::new(U384::from_hex_unchecked(
            "03567bc5ef9c690c2ab2ecdf6a96ef1c139cc0b2f284dca0a9a7943388a49a3aee664ba5379a7655d3c68900be2f6903",
        ));
        let expected_y = FpElement::new(U384::from_hex_unchecked(
            "0b9c15f3fe6e5cf4211f346271d7b01c8f3b28be689c8429c85b67af215533311f0b8dfaaa154fa6b88176c229f2885d",
        ));

        assert_eq!(*px, expected_x, "G1 abc: x mismatch");
        assert_eq!(*py, expected_y, "G1 abc: y mismatch");
    }

    #[test]
    fn test_rfc9380_g2_empty_message() {
        let p = hash_to_g2(b"", RFC_DST_G2).unwrap().to_affine();
        let [px, py, _] = p.coordinates();
        let [px0, px1] = px.value();
        let [py0, py1] = py.value();

        let expected_x0 = FpElement::new(U384::from_hex_unchecked(
            "0141ebfbdca40eb85b87142e130ab689c673cf60f1a3e98d69335266f30d9b8d4ac44c1038e9dcdd5393faf5c41fb78a",
        ));
        let expected_x1 = FpElement::new(U384::from_hex_unchecked(
            "05cb8437535e20ecffaef7752baddf98034139c38452458baeefab379ba13dff5bf5dd71b72418717047f5b0f37da03d",
        ));
        let expected_y0 = FpElement::new(U384::from_hex_unchecked(
            "0503921d7f6a12805e72940b963c0cf3471c7b2a524950ca195d11062ee75ec076daf2d4bc358c4b190c0c98064fdd92",
        ));
        let expected_y1 = FpElement::new(U384::from_hex_unchecked(
            "12424ac32561493f3fe3c260708a12b7c620e7be00099a974e259ddc7d1f6395c3c811cdd19f1e8dbf3e9ecfdcbab8d6",
        ));

        assert_eq!(*px0, expected_x0, "G2 empty msg: x0 mismatch");
        assert_eq!(*px1, expected_x1, "G2 empty msg: x1 mismatch");
        assert_eq!(*py0, expected_y0, "G2 empty msg: y0 mismatch");
        assert_eq!(*py1, expected_y1, "G2 empty msg: y1 mismatch");
    }

    #[test]
    fn test_rfc9380_g2_abc() {
        let p = hash_to_g2(b"abc", RFC_DST_G2).unwrap().to_affine();
        let [px, py, _] = p.coordinates();
        let [px0, px1] = px.value();
        let [py0, py1] = py.value();

        let expected_x0 = FpElement::new(U384::from_hex_unchecked(
            "02c2d18e033b960562aae3cab37a27ce00d80ccd5ba4b7fe0e7a210245129dbec7780ccc7954725f4168aff2787776e6",
        ));
        let expected_x1 = FpElement::new(U384::from_hex_unchecked(
            "139cddbccdc5e91b9623efd38c49f81a6f83f175e80b06fc374de9eb4b41dfe4ca3a230ed250fbe3a2acf73a41177fd8",
        ));
        let expected_y0 = FpElement::new(U384::from_hex_unchecked(
            "1787327b68159716a37440985269cf584bcb1e621d3a7202be6ea05c4cfe244aeb197642555a0645fb87bf7466b2ba48",
        ));
        let expected_y1 = FpElement::new(U384::from_hex_unchecked(
            "00aa65dae3c8d732d10ecd2c50f8a1baf3001578f71c694e03866e9f3d49ac1e1ce70dd94a733534f106d4cec0eddd16",
        ));

        assert_eq!(*px0, expected_x0, "G2 abc: x0 mismatch");
        assert_eq!(*px1, expected_x1, "G2 abc: x1 mismatch");
        assert_eq!(*py0, expected_y0, "G2 abc: y0 mismatch");
        assert_eq!(*py1, expected_y1, "G2 abc: y1 mismatch");
    }

    // ====================================================================
    // Property-based tests
    // ====================================================================

    #[test]
    fn test_hash_to_g1_deterministic() {
        let p1 = hash_to_g1(b"test message", DEFAULT_DST_G1).unwrap();
        let p2 = hash_to_g1(b"test message", DEFAULT_DST_G1).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_hash_to_g1_different_messages() {
        let p1 = hash_to_g1(b"message 1", DEFAULT_DST_G1).unwrap();
        let p2 = hash_to_g1(b"message 2", DEFAULT_DST_G1).unwrap();
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_hash_to_g1_not_identity() {
        let p = hash_to_g1(b"test", DEFAULT_DST_G1).unwrap();
        assert!(!p.is_neutral_element());
    }

    #[test]
    fn test_hash_to_g1_in_subgroup() {
        let p = hash_to_g1(b"subgroup test", DEFAULT_DST_G1).unwrap();
        assert!(p.is_in_subgroup());
    }

    #[test]
    fn test_hash_to_g2_deterministic() {
        let p1 = hash_to_g2(b"test message", DEFAULT_DST_G2).unwrap();
        let p2 = hash_to_g2(b"test message", DEFAULT_DST_G2).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_hash_to_g2_different_messages() {
        let p1 = hash_to_g2(b"message 1", DEFAULT_DST_G2).unwrap();
        let p2 = hash_to_g2(b"message 2", DEFAULT_DST_G2).unwrap();
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_hash_to_g2_not_identity() {
        let p = hash_to_g2(b"test", DEFAULT_DST_G2).unwrap();
        assert!(!p.is_neutral_element());
    }

    #[test]
    fn test_hash_to_g2_in_subgroup() {
        let p = hash_to_g2(b"subgroup test", DEFAULT_DST_G2).unwrap();
        assert!(p.is_in_subgroup());
    }

    #[test]
    fn test_different_dst_produces_different_points() {
        let msg = b"same message";
        let p1 = hash_to_g2(msg, b"DST_ONE").unwrap();
        let p2 = hash_to_g2(msg, b"DST_TWO").unwrap();
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_sswu_g1_produces_valid_point_on_isogenous_curve() {
        let u = FpElement::from(42u64);
        let (x, y) = map_to_curve_sswu_g1(&u);
        let lhs = &y * &y;
        let rhs = &x * &x * &x + &SSWU_A_G1 * &x + &SSWU_B_G1;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_sswu_g2_produces_valid_point_on_isogenous_curve() {
        let u = Fp2Element::new([FpElement::from(42u64), FpElement::from(17u64)]);
        let (x, y) = map_to_curve_sswu_g2(&u);
        let lhs = &y * &y;
        let rhs = &x * &x * &x + &SSWU_A_G2 * &x + &SSWU_B_G2;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_iso_map_g1_produces_point_on_curve() {
        let u = FpElement::from(42u64);
        let (qx, qy) = map_to_curve_sswu_g1(&u);
        let (rx, ry) = iso_map_g1(&qx, &qy);
        let b = BLS12381Curve::b();
        let lhs = &ry * &ry;
        let rhs = &rx * &rx * &rx + &b;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_iso_map_g2_produces_point_on_twist_curve() {
        let u = Fp2Element::new([FpElement::from(42u64), FpElement::from(17u64)]);
        let (qx, qy) = map_to_curve_sswu_g2(&u);
        let (rx, ry) = iso_map_g2(&qx, &qy);
        let b = BLS12381TwistCurve::b();
        let lhs = &ry * &ry;
        let rhs = &rx * &rx * &rx + &b;
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_sswu_g1_sign_consistency() {
        let u = FpElement::from(7u64);
        let (_, y) = map_to_curve_sswu_g1(&u);
        assert_eq!(sgn0_fp(&u), sgn0_fp(&y));
    }

    #[test]
    fn test_sswu_g2_sign_consistency() {
        let u = Fp2Element::new([FpElement::from(7u64), FpElement::from(13u64)]);
        let (_, y) = map_to_curve_sswu_g2(&u);
        assert_eq!(sgn0_fp2(&u), sgn0_fp2(&y));
    }

    #[test]
    fn test_multiple_hashes_g1_all_in_subgroup() {
        for i in 0u64..5 {
            let msg = alloc::format!("test message {}", i);
            let p = hash_to_g1(msg.as_bytes(), DEFAULT_DST_G1).unwrap();
            assert!(p.is_in_subgroup());
        }
    }

    #[test]
    fn test_multiple_hashes_g2_all_in_subgroup() {
        for i in 0u64..5 {
            let msg = alloc::format!("test message {}", i);
            let p = hash_to_g2(msg.as_bytes(), DEFAULT_DST_G2).unwrap();
            assert!(p.is_in_subgroup());
        }
    }
}
