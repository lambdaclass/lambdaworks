use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381FieldModulus;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement as FE, FrField as FrF};
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::cyclic_group::IsGroup;
use num_traits::Num;
use num_bigint::{BigUint, RandBigInt};

use crate::commitments::ipa::traits::IsCommitmentSchemeIPA;

type BaseField = BLS12381PrimeField;
pub type FrElement = FE;
pub type FrField = FrF;

#[allow(non_snake_case)]
pub struct IPA<BLS12381Curve: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve> {
    d: u32,
    H: ShortWeierstrassProjectivePoint<BLS12381Curve>,
    Gs: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>>,    
}

#[allow(non_snake_case)]
pub struct Proof<BLS12381Curve: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve> {
    a: FrElement,
    l: Vec<FrElement>,
    r: Vec<FrElement>,
    L: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>>,
    R: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>>,
}

#[allow(non_snake_case)]
#[allow(clippy::many_single_char_names)]
impl IPA<BLS12381Curve> {
    
    pub fn new(d: u32) -> IPA<BLS12381Curve> {        
        
        let mut gs: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = Vec::new();
        for _ in 0..d {            
            gs.push(generate_random_point::<BLS12381Curve>());
        }

        IPA {
            d,            
            H: generate_random_point::<BLS12381Curve>(),
            Gs: gs,            
        }
    }

}

#[allow(non_snake_case)]
#[allow(clippy::many_single_char_names)]
impl IsCommitmentSchemeIPA for IPA<BLS12381Curve> {

    type CommitmentIPA = Result<ShortWeierstrassProjectivePoint<BLS12381Curve>, String>;
    type OpenIPA = Result<Proof<BLS12381Curve>, String>;
    type VerifyOpen = Result<bool, String>;

    fn commit<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve>(
            &self,
            a: &Polynomial<FrElement>,
            r: FrElement,
        ) -> Self::CommitmentIPA {
            Ok((inner_product_point::<E>(a, &self.Gs)?).operate_with(
                &(self.H.operate_with_self(r.representative())),
            ))
        }    

    fn open(
        &mut self,
        a: &[FrElement],
        b: &[FrElement],
        u: &[FrElement],
        U: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    ) -> Self::OpenIPA {
        let mut a = a.to_owned();
        let mut b = b.to_owned();
        let mut G = self.Gs.clone();

        let k = (f64::from(self.d as u32).log2()) as usize;
        let mut l: Vec<FrElement> = vec![FrElement::zero(); k];
        let mut r: Vec<FrElement> = vec![FrElement::zero(); k];
        let mut L: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = vec![identity_point(); k];
        let mut R: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = vec![identity_point(); k];

        for j in (0..k).rev() {
            let m = a.len() / 2;
            let a_lo = a[..m].to_vec();
            let a_hi = a[m..].to_vec();
            let b_lo = b[..m].to_vec();
            let b_hi = b[m..].to_vec();
            let G_lo = G[..m].to_vec();
            let G_hi = G[m..].to_vec();

            l[j] = generate_random_element();
            r[j] = generate_random_element();

            L[j] = ((inner_product_point::<BLS12381Curve>(&Polynomial::new(&a_lo), &G_hi)?).operate_with(&(self.H.operate_with_self(l[j].representative())))).operate_with(&(U.operate_with_self(inner_product_field(&a_lo, &b_hi)?.representative())));
            R[j] = ((inner_product_point::<BLS12381Curve>(&Polynomial::new(&a_hi), &G_lo)?).operate_with(&(self.H.operate_with_self(r[j].representative())))).operate_with(&(U.operate_with_self(inner_product_field(&a_hi, &b_lo)?.representative())));

            let uj = &u[j];
            let uj_inv = u[j].inv().unwrap();

            a = vec_add(
                &vec_scalar_mul_field(&a_lo, &uj),
                &vec_scalar_mul_field(&a_hi, &uj_inv),
            )?;
            b = vec_add(
                &vec_scalar_mul_field(&b_lo, &uj_inv),
                &vec_scalar_mul_field(&b_hi, &uj),
            )?;            
            G = vec_add_point::<BLS12381Curve>(                
                &vec_scalar_mul_point::<BLS12381Curve>(&G_lo, &uj_inv),
                &vec_scalar_mul_point::<BLS12381Curve>(&G_hi, &uj),
            )?;
        }

        if a.len() != 1 {
            return Err(format!("a.len() should be 1, a.len()={}", a.len()));
        }
        if b.len() != 1 {
            return Err(format!("b.len() should be 1, b.len()={}", b.len()));
        }
        if G.len() != 1 {
            return Err(format!("G.len() should be 1, G.len()={}", G.len()));
        }

        Ok(Proof {
            a: a[0].clone(),
            l,
            r,
            L,
            R,
        })
    } 

    fn verify_open(
        &self,
        x: &FrElement,
        v: &FrElement,
        P: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
        p: &Proof<BLS12381Curve>,
        r: &FrElement,
        u: &[FrElement],
        U: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    ) -> Self::VerifyOpen {
        let P = (*P).operate_with(&(U.operate_with_self(v.representative())));
    
        let mut q_0 = P;
        let mut r = r.clone();
    
        let s = build_s(u, self.d as usize);
        let bs = powers_of(x, self.d.try_into().unwrap());
        let b = inner_product_field(&s, &bs)?;
        let G = inner_product_point::<BLS12381Curve>(&Polynomial::new(&s), &self.Gs)?;
    
        for j in 0..u.len() {
            let uj2 = u[j].square();
            let uj_inv2 = u[j].inv().unwrap().square();
    
            q_0 = (q_0).operate_with(&(p.L[j].operate_with_self(uj2.representative()))).operate_with(&(p.R[j].operate_with_self(uj_inv2.representative())));
            r = r + p.l[j].clone() * uj2 + p.r[j].clone() * uj_inv2;
        }
    
        let q_1 = (G.operate_with_self(p.a.clone().representative())).operate_with(&(self.H.operate_with_self((r).representative()))).operate_with(&(U.operate_with_self((&p.a * b).representative())));
        Ok(q_0 == q_1)
    }

} 

fn build_s(
    u: &[FrElement],
    d: usize,
) -> Vec<FrElement> {
    let k = (f64::from(d as u32).log2()) as usize;
    let mut s: Vec<FrElement> = vec![FrElement::one(); d];
    let mut t = d;
    for j in (0..k).rev() {
        t /= 2;
        let mut c = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..d {
            if c < t {                
                s[i] = s[i].clone() * u[j].clone().inv().unwrap();
            } else {                
                s[i] = s[i].clone() * u[j].clone();
            }
            c += 1;
            if c >= t * 2 {
                c = 0;
            }
        }
    }
    s
}

fn inner_product_field<F: lambdaworks_math::field::traits::IsField>(
    a: &[FieldElement<F>],
    b: &[FieldElement<F>],
) -> Result<FieldElement<F>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "a.len()={} must be equal to b.len()={}",
            a.len(),
            b.len()
        ));
    }
    let mut c: FieldElement<F> = FieldElement::<F>::zero();
    for i in 0..a.len() {
        c += a[i].clone() * b[i].clone();
    }
    Ok(c)
}

fn inner_product_point<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve>(
    a: &Polynomial<FrElement>,
    b: &[ShortWeierstrassProjectivePoint<BLS12381Curve>],
) -> Result<ShortWeierstrassProjectivePoint<BLS12381Curve>, String> {
    
    let coef = &a.coefficients;

    if coef.len() != b.len() {
        return Err(format!(
            "coef.len()={} must be equal to b.len()={}",
            coef.len(),
            b.len()
        ));
    }
    
    let mut c: ShortWeierstrassProjectivePoint<BLS12381Curve> = identity_point();
    for i in 0..coef.len() {        
        let d = b[i].operate_with_self(coef[i].representative());
        c = c.operate_with(&d);
    }
    Ok(c)
}

fn vec_add(
    a: &[FrElement],
    b: &[FrElement],
) -> Result<Vec<FrElement>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "a.len()={} must be equal to b.len()={}",
            a.len(),
            b.len()
        ));
    }
    let mut c: Vec<FrElement> = vec![FrElement::zero(); a.len()];
    for i in 0..a.len() {
        c[i] = a[i].clone() + b[i].clone();
    }
    Ok(c)
}

fn vec_add_point<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve>(
    a: &[ShortWeierstrassProjectivePoint<BLS12381Curve>],
    b: &[ShortWeierstrassProjectivePoint<BLS12381Curve>],
) -> Result<Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "a.len()={} must be equal to b.len()={}",
            a.len(),
            b.len()
        ));
    }
    let mut c: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = vec![identity_point(); a.len()];
    for i in 0..a.len() {        
        c[i] = a[i].operate_with(&b[i]);
    }
    Ok(c)
}

fn vec_scalar_mul_field(
    a: &[FrElement],
    b: &FrElement,
) -> Vec<FrElement> {
    let mut c: Vec<FrElement> = vec![FrElement::zero(); a.len()];
    for i in 0..a.len() {
        c[i] = a[i].clone() * b;
    }
    c
}
fn vec_scalar_mul_point<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve>(
    a: &[ShortWeierstrassProjectivePoint<BLS12381Curve>],
    b: &FrElement,
) -> Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> {
    let mut c: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = vec![identity_point(); a.len()];
    for i in 0..a.len() {
        c[i] = a[i].operate_with_self(b.representative());
    }
    c
}

fn powers_of(
    x: &FrElement,
    d: usize,
) -> Vec<FrElement> {
    let mut c: Vec<FrElement> = vec![FrElement::zero(); d as usize];
    c[0] = x.clone();
    for i in 1..d as usize {
        c[i] = c[i - 1].clone() * x.clone();
    }
    c
}

pub fn identity_point<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve<BaseField = MontgomeryBackendPrimeField<BLS12381FieldModulus, 6>> + lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass>() -> ShortWeierstrassProjectivePoint<E> {
    let point = ShortWeierstrassProjectivePoint::new([
        FieldElement::<BaseField>::new_base("0"),
        FieldElement::<BaseField>::new_base("1"),
        FieldElement::<BaseField>::new_base("0"),
    ]);
    point
}

pub fn generate_random_element() -> FrElement {
    let max_value_str = "1000000000000000000000000000000000000000000000000000000000000000000000000000"; // Fix this value!
    let max_value = BigUint::from_str_radix(max_value_str, 10).unwrap();
    let mut rng = rand::thread_rng();
    let max_bits = max_value.bits();
    let mut random_biguint;
    loop {        
        random_biguint = rng.gen_biguint(max_bits);
        if random_biguint <= max_value {
            break;
        }
    }
    let element: FrElement = FrElement::new(UnsignedInteger::from_hex_unchecked(
        &random_biguint.to_str_radix(16),
    ));
    element
}

pub fn generate_random_point<E: lambdaworks_math::elliptic_curve::traits::IsEllipticCurve>() -> ShortWeierstrassProjectivePoint<BLS12381Curve> {
    let _g_1: ShortWeierstrassProjectivePoint<BLS12381Curve> = BLS12381Curve::generator();
    let point = _g_1.operate_with_self(generate_random_element().representative());
    point
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use crate::commitments::ipa::traits::IsCommitmentSchemeIPA;
    use super::*;
    use lambdaworks_math::unsigned_integer::element::U256;

    #[test]
    fn identity_point_function_works() {
        let gen: ShortWeierstrassProjectivePoint<BLS12381Curve> = BLS12381Curve::generator();

        let G_5 = gen.operate_with_self(5_u64);
        let G_7 = gen.operate_with_self(7_u64);
        let G_9 = gen.operate_with_self(9_u64);

        let adding_G_5_with_identity_point = G_5.operate_with(&identity_point());
        let adding_G_7_with_identity_point = G_7.operate_with(&identity_point());
        let adding_G_9_with_identity_point = G_9.operate_with(&identity_point());

        assert_eq!(G_5, adding_G_5_with_identity_point);
        assert_eq!(G_7, adding_G_7_with_identity_point);
        assert_eq!(G_9, adding_G_9_with_identity_point);
    }

    #[test]
    fn inner_product_field_works() {
        let a = vec![
            FrElement::from(1 as u64),
            FrElement::from(2 as u64),
            FrElement::from(3 as u64),
            FrElement::from(4 as u64),
        ];
        let b = vec![
            FrElement::from(1 as u64),
            FrElement::from(2 as u64),
            FrElement::from(3 as u64),
            FrElement::from(4 as u64),
        ];
        let c = inner_product_field(&a, &b).unwrap();
        assert_eq!(c, FrElement::from(30 as u64));
    }    

    #[test]
    fn inner_product_point_works() {
        let gen: ShortWeierstrassProjectivePoint<BLS12381Curve> = BLS12381Curve::generator();        

        let a = 
            Polynomial::new(&[
                FE::new(U256::from_u64(1)),
                FE::new(U256::from_u64(2)),
                FE::new(U256::from_u64(3)),
                FE::new(U256::from_u64(4))]);

        let coef = a.clone().coefficients;

        let b = vec![
            gen.operate_with_self(5_u64),
            gen.operate_with_self(6_u64),
            gen.operate_with_self(7_u64),
            gen.operate_with_self(8_u64),
        ];

        let c = inner_product_point::<BLS12381Curve>(&a, &b).unwrap();
        let c_expected = 
            (((b[0].operate_with_self(coef[0].representative())).operate_with(&(b[1].operate_with_self(coef[1].representative())))).operate_with(&(b[2].operate_with_self(coef[2].representative())))).operate_with(&(b[3].operate_with_self(coef[3].representative())));

        assert_eq!(c, c_expected);
    }    
   
    #[test]
    fn test_homomorphic_property() {
        let d = 8;
        let ipa = IPA::new(d);

        let a = 
            Polynomial::new(&[                
                FrElement::from(1 as u64),
                FrElement::from(2 as u64),
                FrElement::from(3 as u64),
                FrElement::from(4 as u64),
                FrElement::from(5 as u64),
                FrElement::from(6 as u64),
                FrElement::from(7 as u64),
                FrElement::from(8 as u64)]);

        let b = a.clone();

        let coef_a = a.clone().coefficients;
        let coef_b = coef_a.clone();

        let r = generate_random_element();
        let s = generate_random_element();

        let vc_a = ipa.commit::<BLS12381Curve>(&a, r.clone()).unwrap();
        let vc_b = ipa.commit::<BLS12381Curve>(&b, s.clone()).unwrap();

        let expected_vc_c = ipa.commit::<BLS12381Curve>(&Polynomial::new(&vec_add(&coef_a, &coef_b).unwrap()), r + s).unwrap();
        let vc_c = vc_a.operate_with(&vc_b);
        assert_eq!(vc_c, expected_vc_c);
    }    
  
    #[test]
    fn test_inner_product_argument_proof() {
        let d = 8;
        let mut ipa = IPA::new(d);

        let a = 
            Polynomial::new(&[                
                FrElement::from(1 as u64),
                FrElement::from(2 as u64),
                FrElement::from(3 as u64),
                FrElement::from(4 as u64),
                FrElement::from(5 as u64),
                FrElement::from(6 as u64),
                FrElement::from(7 as u64),
                FrElement::from(8 as u64)]);

        let coef_a = a.clone().coefficients;

        let r = generate_random_element();

        // Prover commits
        let P = ipa.commit::<BLS12381Curve>(&a, r.clone()).unwrap();

        // Verifier sets challenges
        let U = generate_random_point::<BLS12381Curve>();
        let k = (f64::from(ipa.d as u32).log2()) as usize;
        let mut u: Vec<FrElement> = vec![FrElement::zero(); k];
        for j in 0..k {
            u[j] = generate_random_element();
        }
        let x = FrElement::from(3 as u64);

        // Prover opens at the challenges
        let b = powers_of(&x.clone(), ipa.d.try_into().unwrap());
        let v = inner_product_field(&coef_a, &b).unwrap();
        let proof = ipa.open(&coef_a, &b, &u, &U).unwrap();

        // Verifier
        let verif = ipa.verify_open(&x, &v, &P, &proof, &r, &u, &U).unwrap();
        assert!(verif);
    }

}
