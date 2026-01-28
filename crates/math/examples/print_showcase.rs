use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            field_extension::{
                BLS12381PrimeField, Degree2ExtensionField as BLS12381Degree2ExtensionField,
            },
        },
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve,
            field_extension::{BN254PrimeField, Degree2ExtensionField},
        },
        traits::IsEllipticCurve,
    },
    field::{
        element::FieldElement,
        fields::fft_friendly::{
            babybear::Babybear31PrimeField, quartic_babybear::Degree4BabyBearExtensionField,
            stark_252_prime_field::Stark252PrimeField,
        },
    },
};

fn main() {
    type BabyBear = FieldElement<Babybear31PrimeField>;
    type BabyBear4 = FieldElement<Degree4BabyBearExtensionField>;
    type BnFp = FieldElement<BN254PrimeField>;
    type BnFp2 = FieldElement<Degree2ExtensionField>;
    type Stark = FieldElement<Stark252PrimeField>;
    type BlsFp = FieldElement<BLS12381PrimeField>;
    type BlsFp2 = FieldElement<BLS12381Degree2ExtensionField>;

    let a = BabyBear::from(42);
    let b = BabyBear::from(2013265920u64); // -1 mod p
    println!("BabyBear:");
    println!("  a = {a}");
    println!("  b = {b}");
    println!("  a + b = {}", &a + &b);
    println!("  a * b = {}", &a * &b);

    let x = BnFp::from_hex_unchecked("0x1234abcd");
    let y = BnFp::from_hex_unchecked("0xdeadbeef");
    println!();
    println!("BN254 base field:");
    println!("  x = {x}");
    println!("  y = {y}");
    println!("  x + y = {}", &x + &y);

    let fp2 = BnFp2::new([BnFp::from(5), BnFp::from(7)]);
    println!();
    println!("BN254 Fp2 element:");
    println!("  fp2 = {fp2}");
    println!("  fp2^2 = {}", fp2.square());

    let bb4 = BabyBear4::new([
        BabyBear::from(1),
        BabyBear::from(2),
        BabyBear::from(3),
        BabyBear::from(4),
    ]);
    println!();
    println!("BabyBear degree-4 extension:");
    println!("  e = {bb4}");
    println!("  e^2 = {}", bb4.square());

    let g = BN254Curve::generator();
    let g2 = g.operate_with_self(2u64);
    println!();
    println!("BN254 curve points:");
    println!("  G:\n{g}");
    println!("  2G (projective):\n{g2}");
    println!("  2G (affine):\n{}", g2.to_affine());

    let s = Stark::from_hex_unchecked("0x1234");
    println!();
    println!("Stark252 field:");
    println!("  s = {s}");
    println!("  s^2 = {}", s.square());

    let bls_fp2 = BlsFp2::new([BlsFp::from(9), BlsFp::from(10)]);
    println!();
    println!("BLS12-381 Fp2 element:");
    println!("  bls_fp2 = {bls_fp2}");
    println!("  bls_fp2^2 = {}", bls_fp2.square());

    let gj = BLS12381Curve::generator();
    let gj2 = gj.operate_with_self(2u64);
    println!();
    println!("BLS12-381 curve points (Jacobian):");
    println!("  G:\n{gj}");
    println!("  2G (jacobian):\n{gj2}");
    println!("  2G (affine):\n{}", gj2.to_affine());
}
