use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

fn main() {
    println!("=== FIELD ELEMENTS ===\n");
    
    // Stark252 Field
    println!("--- Stark252PrimeField ---");
    let zero = FieldElement::<Stark252PrimeField>::from(0u64);
    let one = FieldElement::<Stark252PrimeField>::from(1u64);
    let small = FieldElement::<Stark252PrimeField>::from(12345u64);
    let medium = FieldElement::<Stark252PrimeField>::from(0xdeadbeef_u64);
    let large = FieldElement::<Stark252PrimeField>::from(u64::MAX);
    
    println!("zero:   {}", zero);
    println!("one:    {}", one);
    println!("small:  {} (12345)", small);
    println!("medium: {} (0xdeadbeef)", medium);
    println!("large:  {} (u64::MAX)", large);
    println!();
    
    // Goldilocks Field  
    println!("--- Goldilocks64Field ---");
    let g_zero = FieldElement::<Goldilocks64Field>::from(0u64);
    let g_one = FieldElement::<Goldilocks64Field>::from(1u64);
    let g_small = FieldElement::<Goldilocks64Field>::from(42u64);
    let g_large = FieldElement::<Goldilocks64Field>::from(1000000007u64);
    
    println!("zero:   {:?}", g_zero);
    println!("one:    {:?}", g_one);
    println!("small:  {:?} (42)", g_small);
    println!("large:  {:?} (1000000007)", g_large);
    println!();
    
    // Mersenne31 Field
    println!("--- Mersenne31Field ---");
    let m_zero = FieldElement::<Mersenne31Field>::from(0u64);
    let m_one = FieldElement::<Mersenne31Field>::from(1u64);
    let m_small = FieldElement::<Mersenne31Field>::from(255u64);
    let m_large = FieldElement::<Mersenne31Field>::from(123456789u64);
    
    println!("zero:   {:?}", m_zero);
    println!("one:    {:?}", m_one);  
    println!("small:  {:?} (255)", m_small);
    println!("large:  {:?} (123456789)", m_large);
    println!();
    
    println!("=== ELLIPTIC CURVE POINTS ===\n");
    
    // BLS12-381
    println!("--- BLS12-381 Generator ---");
    let g1 = BLS12381Curve::generator();
    println!("{:?}", g1);
    println!();
    
    // BN254
    println!("--- BN254 Generator ---");
    let bn_g = BN254Curve::generator();
    println!("{:?}", bn_g);
    println!();
    
    // Point at infinity
    println!("--- Point at Infinity (BLS12-381) ---");
    let inf = BLS12381Curve::generator().operate_with_self(0u64);
    println!("{:?}", inf);
    println!();
    
    // Projective point (z != 1)
    println!("--- Projective Point (z != 1) ---");
    let g2 = BLS12381Curve::generator();
    let doubled = g2.operate_with(&g2);  // This might have z != 1 before normalization
    println!("{:?}", doubled);
}
