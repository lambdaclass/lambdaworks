use crate::alloc::borrow::ToOwned;
use crate::commitments::ipa::traits::IsCommitmentSchemeIPA;
use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::transcript::Transcript;
use alloc::string::String;
use alloc::vec::Vec;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use lambdaworks_math::elliptic_curve::short_weierstrass::traits::IsShortWeierstrass;
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::traits::IsPrimeField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use rand::Rng;
use std::marker::PhantomData;

pub struct IPA<F: IsField + IsPrimeField, E: IsEllipticCurve + IsShortWeierstrass> {
    d: u32,
    _h: ShortWeierstrassProjectivePoint<E>,
    _gs: Vec<ShortWeierstrassProjectivePoint<E>>,
    _field_type: PhantomData<F>,
}

pub struct Proof<F: IsField + IsPrimeField, E: IsEllipticCurve + IsShortWeierstrass> {
    a: FieldElement<F>,
    l: Vec<FieldElement<F>>,
    r: Vec<FieldElement<F>>,
    _l: Vec<ShortWeierstrassProjectivePoint<E>>,
    _r: Vec<ShortWeierstrassProjectivePoint<E>>,
}

pub struct StructuredReferenceString<E: IsEllipticCurve + IsShortWeierstrass> {
    _h: ShortWeierstrassProjectivePoint<E>,
    _gs: Vec<ShortWeierstrassProjectivePoint<E>>,
}

impl<E: IsEllipticCurve + IsShortWeierstrass> StructuredReferenceString<E> {
    pub fn new<F: IsField<BaseType = UnsignedInteger<M>> + IsPrimeField, const M: usize>(
        d: u32,
        gen: ShortWeierstrassProjectivePoint<E>,
    ) -> StructuredReferenceString<E> {
        let mut gs: Vec<ShortWeierstrassProjectivePoint<E>> = Vec::new();
        for _ in 0..d {
            gs.push(generate_random_point::<F, M, E>(gen.clone()));
        }

        StructuredReferenceString::<E> {
            _h: generate_random_point::<F, M, E>(gen),
            _gs: gs,
        }
    }
}

impl<
        F: IsField<BaseType = UnsignedInteger<M>> + IsPrimeField,
        E: IsEllipticCurve + IsShortWeierstrass,
        const M: usize,
    > IPA<F, E>
{
    pub fn new(d: u32, gen: ShortWeierstrassProjectivePoint<E>) -> IPA<F, E> {
        let ref_str = StructuredReferenceString::<E>::new::<F, M>(d, gen);

        IPA::<F, E> {
            d,
            _h: ref_str._h,
            _gs: ref_str._gs,
            _field_type: PhantomData,
        }
    }
}

impl<
        F: IsField<BaseType = UnsignedInteger<M>> + IsPrimeField,
        E: IsEllipticCurve + IsShortWeierstrass,
        const M: usize,
    > IsCommitmentSchemeIPA<F, E> for IPA<F, E>
{
    type CommitmentIPA = Result<ShortWeierstrassProjectivePoint<E>, String>;
    type OpenIPA = Result<Proof<F, E>, String>;
    type VerifyOpen = Result<bool, String>;

    fn commit(
        &self,
        a: &Polynomial<FieldElement<F>>,
        r: FieldElement<F>,
        neutral_element: ShortWeierstrassProjectivePoint<E>,
    ) -> Self::CommitmentIPA {
        Ok((inner_product_point(a, &self._gs, neutral_element)?)
            .operate_with(&(self._h.operate_with_self(r.representative()))))
    }

    fn open(
        &mut self,
        a: &[FieldElement<F>],
        b: &[FieldElement<F>],
        u: &[FieldElement<F>],
        u_point: &ShortWeierstrassProjectivePoint<E>,
        neutral_element: ShortWeierstrassProjectivePoint<E>,
    ) -> Self::OpenIPA {
        let mut a = a.to_owned();
        let mut b = b.to_owned();
        let mut _g = self._gs.clone();

        let k = (f64::from(self.d).log2()) as usize;
        let mut l: Vec<FieldElement<F>> = vec![FieldElement::<F>::zero(); k];
        let mut r: Vec<FieldElement<F>> = vec![FieldElement::<F>::zero(); k];
        let mut _l: Vec<ShortWeierstrassProjectivePoint<E>> = vec![neutral_element.clone(); k];
        let mut _r: Vec<ShortWeierstrassProjectivePoint<E>> = vec![neutral_element.clone(); k];

        for j in (0..k).rev() {
            let m = a.len() / 2;
            let a_lo: Vec<_> = a.iter().take(m).cloned().collect();
            let a_hi: Vec<_> = a.iter().skip(m).cloned().collect();
            let b_lo: Vec<_> = b.iter().take(m).cloned().collect();
            let b_hi: Vec<_> = b.iter().skip(m).cloned().collect();
            let _g_lo: Vec<_> = _g.iter().take(m).cloned().collect();
            let _g_hi: Vec<_> = _g.iter().skip(m).cloned().collect();

            l[j] = generate_random_element();
            r[j] = generate_random_element();

            _l[j] = ((inner_product_point::<F, E>(
                &Polynomial::new(&a_lo),
                &_g_hi,
                neutral_element.clone(),
            )?)
            .operate_with(&(self._h.operate_with_self(l[j].representative()))))
            .operate_with(
                &(u_point.operate_with_self(inner_product_field(&a_lo, &b_hi)?.representative())),
            );
            _r[j] = ((inner_product_point::<F, E>(
                &Polynomial::new(&a_hi),
                &_g_lo,
                neutral_element.clone(),
            )?)
            .operate_with(&(self._h.operate_with_self(r[j].representative()))))
            .operate_with(
                &(u_point.operate_with_self(inner_product_field(&a_hi, &b_lo)?.representative())),
            );

            let uj = &u[j];
            let uj_inv = u[j].inv().unwrap();

            a = vec_add(
                &vec_scalar_mul_field(&a_lo, uj),
                &vec_scalar_mul_field(&a_hi, &uj_inv),
            )?;
            b = vec_add(
                &vec_scalar_mul_field(&b_lo, &uj_inv),
                &vec_scalar_mul_field(&b_hi, uj),
            )?;
            _g = vec_add_point::<E>(
                &vec_scalar_mul_point::<F, E>(&_g_lo, &uj_inv, neutral_element.clone()),
                &vec_scalar_mul_point::<F, E>(&_g_hi, uj, neutral_element.clone()),
                neutral_element.clone(),
            )?;
        }

        if a.len() != 1 {
            return Err(format!("a.len() should be 1, a.len()={}", a.len()));
        }
        if b.len() != 1 {
            return Err(format!("b.len() should be 1, b.len()={}", b.len()));
        }
        if _g.len() != 1 {
            return Err(format!("G.len() should be 1, G.len()={}", _g.len()));
        }

        Ok(Proof {
            a: a[0].clone(),
            l,
            r,
            _l,
            _r,
        })
    }

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
    ) -> Self::VerifyOpen {
        let _p = (*_p).operate_with(&(u_point.operate_with_self(v.representative())));

        let mut q_0 = _p;
        let mut r = r.clone();

        let s = build_s(u, self.d as usize);
        let bs = powers_of(x, self.d.try_into().unwrap());
        let b = inner_product_field(&s, &bs)?;
        let _g = inner_product_point::<F, E>(&Polynomial::new(&s), &self._gs, neutral_element)?;

        for (j, item) in u.iter().enumerate() {
            let uj2 = item.square();
            let uj_inv2 = item.inv().unwrap().square();

            q_0 = (q_0)
                .operate_with(&(p._l[j].operate_with_self(uj2.representative())))
                .operate_with(&(p._r[j].operate_with_self(uj_inv2.representative())));
            r = r + p.l[j].clone() * uj2 + p.r[j].clone() * uj_inv2;
        }

        let q_1 = (_g.operate_with_self(p.a.clone().representative()))
            .operate_with(&(self._h.operate_with_self((r).representative())))
            .operate_with(&(u_point.operate_with_self((&p.a * b).representative())));
        Ok(q_0 == q_1)
    }
}

fn build_s<F: IsField>(u: &[FieldElement<F>], d: usize) -> Vec<FieldElement<F>> {
    let k = (f64::from(d as u32).log2()) as usize;
    let mut s: Vec<FieldElement<F>> = vec![FieldElement::<F>::one(); d];
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

fn inner_product_field<F: IsField>(
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

fn inner_product_point<F: IsPrimeField, E: IsEllipticCurve + IsShortWeierstrass>(
    a: &Polynomial<FieldElement<F>>,
    b: &[ShortWeierstrassProjectivePoint<E>],
    neutral_element: ShortWeierstrassProjectivePoint<E>,
) -> Result<ShortWeierstrassProjectivePoint<E>, String> {
    let coef = &a.coefficients;

    if coef.len() != b.len() {
        return Err(format!(
            "coef.len()={} must be equal to b.len()={}",
            coef.len(),
            b.len()
        ));
    }

    let mut c: ShortWeierstrassProjectivePoint<E> = neutral_element;
    for i in 0..coef.len() {
        let d = b[i].operate_with_self(coef[i].representative());
        c = c.operate_with(&d);
    }
    Ok(c)
}

fn vec_add<F: IsField>(
    a: &[FieldElement<F>],
    b: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "a.len()={} must be equal to b.len()={}",
            a.len(),
            b.len()
        ));
    }
    let mut c: Vec<FieldElement<F>> = vec![FieldElement::<F>::zero(); a.len()];
    for i in 0..a.len() {
        c[i] = a[i].clone() + b[i].clone();
    }
    Ok(c)
}

fn vec_add_point<E: IsEllipticCurve + IsShortWeierstrass>(
    a: &[ShortWeierstrassProjectivePoint<E>],
    b: &[ShortWeierstrassProjectivePoint<E>],
    neutral_element: ShortWeierstrassProjectivePoint<E>,
) -> Result<Vec<ShortWeierstrassProjectivePoint<E>>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "a.len()={} must be equal to b.len()={}",
            a.len(),
            b.len()
        ));
    }
    let mut c: Vec<ShortWeierstrassProjectivePoint<E>> = vec![neutral_element; a.len()];
    for i in 0..a.len() {
        c[i] = a[i].operate_with(&b[i]);
    }
    Ok(c)
}

fn vec_scalar_mul_field<F: IsField>(
    a: &[FieldElement<F>],
    b: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let mut c: Vec<FieldElement<F>> = vec![FieldElement::<F>::zero(); a.len()];
    for i in 0..a.len() {
        c[i] = a[i].clone() * b;
    }
    c
}
fn vec_scalar_mul_point<F: IsPrimeField, E: IsEllipticCurve + IsShortWeierstrass>(
    a: &[ShortWeierstrassProjectivePoint<E>],
    b: &FieldElement<F>,
    neutral_element: ShortWeierstrassProjectivePoint<E>,
) -> Vec<ShortWeierstrassProjectivePoint<E>> {
    let mut c: Vec<ShortWeierstrassProjectivePoint<E>> = vec![neutral_element; a.len()];
    for i in 0..a.len() {
        c[i] = a[i].operate_with_self(b.representative());
    }
    c
}

fn powers_of<F: IsField>(x: &FieldElement<F>, d: usize) -> Vec<FieldElement<F>> {
    let mut c: Vec<FieldElement<F>> = vec![FieldElement::zero(); d];
    c[0] = x.clone();
    for i in 1..d {
        c[i] = c[i - 1].clone() * x.clone();
    }
    c
}

pub fn generate_random_element<F: IsField<BaseType = UnsignedInteger<M>>, const M: usize>(
) -> FieldElement<F> {
    let mut rng = rand::thread_rng();
    let mut limbs: [u64; M] = [0; M];
    for limb in &mut limbs {
        *limb = rng.gen::<u64>();
    }
    FieldElement::<F>::new(UnsignedInteger::<M>::from_limbs(limbs))
}

pub fn generate_random_point<
    F: IsField<BaseType = UnsignedInteger<M>>,
    const M: usize,
    E: IsEllipticCurve + IsShortWeierstrass,
>(
    gen: ShortWeierstrassProjectivePoint<E>,
) -> ShortWeierstrassProjectivePoint<E>
where
    F: lambdaworks_math::field::traits::IsPrimeField,
{
    let mut rng = rand::thread_rng();
    let mut limbs: [u64; M] = [0; M];
    for limb in &mut limbs {
        *limb = rng.gen::<u64>();
    }
    gen.operate_with_self(
        FieldElement::<F>::new(UnsignedInteger::<M>::from_limbs(limbs)).representative(),
    )
}

pub fn build_field_challenge<F: IsField<BaseType = UnsignedInteger<M>>, const M: usize>(
    vector: Vec<u8>,
) -> FieldElement<F> {
    let mut transcript = DefaultTranscript::new();
    for element in vector.iter() {
        transcript.append(&[*element]);
    }
    FieldElement::<F>::new(UnsignedInteger::from_bytes_be(&transcript.challenge()).unwrap())
}

pub fn build_group_challenge<
    F: IsField<BaseType = UnsignedInteger<M>>,
    const M: usize,
    E: IsEllipticCurve + IsShortWeierstrass,
>(
    vector: Vec<u8>,
    gen: ShortWeierstrassProjectivePoint<E>,
) -> ShortWeierstrassProjectivePoint<E>
where
    F: lambdaworks_math::field::traits::IsPrimeField,
{
    let mut transcript = DefaultTranscript::new();
    for element in vector.iter() {
        transcript.append(&[*element]);
    }
    gen.operate_with_self(
        FieldElement::<F>::new(UnsignedInteger::from_bytes_be(&transcript.challenge()).unwrap())
            .representative(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitments::ipa::traits::IsCommitmentSchemeIPA;
    use rand::Rng;

    // Tests using BLS12381Curve:
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement as FE_1;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField as Fr_1;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381FieldModulus;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    use lambdaworks_math::field::fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField;
    type BaseField = BLS12381PrimeField;

    pub fn neutral_element_bls12381_curve<
        E: IsEllipticCurve<BaseField = MontgomeryBackendPrimeField<BLS12381FieldModulus, 6>>
            + IsShortWeierstrass,
    >() -> ShortWeierstrassProjectivePoint<E> {
        ShortWeierstrassProjectivePoint::new([
            FieldElement::<BaseField>::new_base("0"),
            FieldElement::<BaseField>::new_base("1"),
            FieldElement::<BaseField>::new_base("0"),
        ])
    }

    #[test]
    fn inner_product_field_works() {
        let a = vec![
            FE_1::from(1_u64),
            FE_1::from(2_u64),
            FE_1::from(3_u64),
            FE_1::from(4_u64),
        ];
        let b = vec![
            FE_1::from(1_u64),
            FE_1::from(2_u64),
            FE_1::from(3_u64),
            FE_1::from(4_u64),
        ];
        let c = inner_product_field(&a, &b).unwrap();
        assert_eq!(c, FE_1::from(30_u64));
    }

    #[test]
    fn neutral_element_bls12381_curve_function_works() {
        let gen = BLS12381Curve::generator();
        let g_5 = gen.operate_with_self(5_u64);
        let g_7 = gen.operate_with_self(7_u64);
        let g_9 = gen.operate_with_self(9_u64);
        let adding_g_5_with_neutral_element = g_5.operate_with(&neutral_element_bls12381_curve());
        let adding_g_7_with_neutral_element = g_7.operate_with(&neutral_element_bls12381_curve());
        let adding_g_9_with_neutral_element = g_9.operate_with(&neutral_element_bls12381_curve());
        assert_eq!(g_5, adding_g_5_with_neutral_element);
        assert_eq!(g_7, adding_g_7_with_neutral_element);
        assert_eq!(g_9, adding_g_9_with_neutral_element);
    }

    #[test]
    fn inner_product_point_works() {
        let gen = BLS12381Curve::generator();
        let a = Polynomial::new(&[
            FE_1::from(1_u64),
            FE_1::from(2_u64),
            FE_1::from(3_u64),
            FE_1::from(4_u64),
        ]);
        let coef = a.clone().coefficients;
        let b = vec![
            gen.operate_with_self(5_u64),
            gen.operate_with_self(6_u64),
            gen.operate_with_self(7_u64),
            gen.operate_with_self(8_u64),
        ];
        let c = inner_product_point(&a, &b, neutral_element_bls12381_curve()).unwrap();
        let c_expected = (((b[0].operate_with_self(coef[0].representative()))
            .operate_with(&(b[1].operate_with_self(coef[1].representative()))))
        .operate_with(&(b[2].operate_with_self(coef[2].representative()))))
        .operate_with(&(b[3].operate_with_self(coef[3].representative())));

        assert_eq!(c, c_expected);
    }

    #[test]
    fn test_homomorphic_property_bls12381_curve() {
        let gen = BLS12381Curve::generator();
        let d = 8;
        let ipa = IPA::new(d, gen);
        let a = Polynomial::new(&[
            FE_1::from(1_u64),
            FE_1::from(2_u64),
            FE_1::from(3_u64),
            FE_1::from(4_u64),
            FE_1::from(5_u64),
            FE_1::from(6_u64),
            FE_1::from(7_u64),
            FE_1::from(8_u64),
        ]);
        let b = a.clone();
        let coef_a = a.clone().coefficients;
        let coef_b = coef_a.clone();
        let r = generate_random_element();
        let s = generate_random_element();
        let vc_a = ipa
            .commit(&a, r.clone(), neutral_element_bls12381_curve())
            .unwrap();
        let vc_b = ipa
            .commit(&b, s.clone(), neutral_element_bls12381_curve())
            .unwrap();
        let expected_vc_c = ipa
            .commit(
                &Polynomial::new(&vec_add(&coef_a, &coef_b).unwrap()),
                r + s,
                neutral_element_bls12381_curve(),
            )
            .unwrap();
        let vc_c = vc_a.operate_with(&vc_b);
        assert_eq!(vc_c, expected_vc_c);
    }

    #[test]
    fn test_inner_product_argument_proof_bls12381_curve() {
        const M: usize = 4;
        let gen = BLS12381Curve::generator();
        let d = 8;
        let mut ipa = IPA::new(d, gen.clone());
        let a = Polynomial::new(&[
            FE_1::from(1_u64),
            FE_1::from(2_u64),
            FE_1::from(3_u64),
            FE_1::from(4_u64),
            FE_1::from(5_u64),
            FE_1::from(6_u64),
            FE_1::from(7_u64),
            FE_1::from(8_u64),
        ]);
        let coef_a = a.clone().coefficients;
        let r = generate_random_element();
        let _p = ipa
            .commit(&a, r.clone(), neutral_element_bls12381_curve())
            .unwrap();
        let mut rng = rand::thread_rng();
        let vector: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        let _u = build_group_challenge::<Fr_1, M, BLS12381Curve>(vector, gen);
        let k = (f64::from(ipa.d).log2()) as usize;
        let mut u: Vec<FE_1> = vec![FE_1::zero(); k];
        for item in u.iter_mut().take(k) {
            let mut rng = rand::thread_rng();
            let vector: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
            *item = build_field_challenge(vector);
        }
        let x = FE_1::from(3_u64);
        let b = powers_of(&x.clone(), ipa.d.try_into().unwrap());
        let v = inner_product_field(&coef_a, &b).unwrap();
        let proof = ipa
            .open(&coef_a, &b, &u, &_u, neutral_element_bls12381_curve())
            .unwrap();
        let verif = ipa
            .verify_open(
                &x,
                &v,
                &_p,
                &proof,
                &r,
                &u,
                &_u,
                neutral_element_bls12381_curve(),
            )
            .unwrap();
        assert!(verif);
    }

    // Tests using BN254Curve:
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement as FE_2;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrField as Fr_2;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254FieldModulus;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField;
    type BaseField2 = BN254PrimeField;

    pub fn neutral_element_bn254_curve<
        E: IsEllipticCurve<BaseField = MontgomeryBackendPrimeField<BN254FieldModulus, 4>>
            + IsShortWeierstrass,
    >() -> ShortWeierstrassProjectivePoint<E> {
        ShortWeierstrassProjectivePoint::new([
            FieldElement::<BaseField2>::new_base("0"),
            FieldElement::<BaseField2>::new_base("1"),
            FieldElement::<BaseField2>::new_base("0"),
        ])
    }

    #[test]
    fn neutral_element_bn254_function_works() {
        let gen = BN254Curve::generator();
        let g_5 = gen.operate_with_self(5_u64);
        let g_7 = gen.operate_with_self(7_u64);
        let g_9 = gen.operate_with_self(9_u64);
        let adding_g_5_with_neutral_element = g_5.operate_with(&neutral_element_bn254_curve());
        let adding_g_7_with_neutral_element = g_7.operate_with(&neutral_element_bn254_curve());
        let adding_g_9_with_neutral_element = g_9.operate_with(&neutral_element_bn254_curve());
        assert_eq!(g_5, adding_g_5_with_neutral_element);
        assert_eq!(g_7, adding_g_7_with_neutral_element);
        assert_eq!(g_9, adding_g_9_with_neutral_element);
    }

    #[test]
    fn test_homomorphic_property_bn254_curve() {
        let gen = BN254Curve::generator();
        let d = 8;
        let ipa = IPA::new(d, gen);
        let a = Polynomial::new(&[
            FE_2::from(1_u64),
            FE_2::from(2_u64),
            FE_2::from(3_u64),
            FE_2::from(4_u64),
            FE_2::from(5_u64),
            FE_2::from(6_u64),
            FE_2::from(7_u64),
            FE_2::from(8_u64),
        ]);
        let b = a.clone();
        let coef_a = a.clone().coefficients;
        let coef_b = coef_a.clone();
        let r = generate_random_element();
        let s = generate_random_element();
        let vc_a = ipa
            .commit(&a, r.clone(), neutral_element_bn254_curve())
            .unwrap();
        let vc_b = ipa
            .commit(&b, s.clone(), neutral_element_bn254_curve())
            .unwrap();
        let expected_vc_c = ipa
            .commit(
                &Polynomial::new(&vec_add(&coef_a, &coef_b).unwrap()),
                r + s,
                neutral_element_bn254_curve(),
            )
            .unwrap();
        let vc_c = vc_a.operate_with(&vc_b);
        assert_eq!(vc_c, expected_vc_c);
    }

    #[test]
    fn test_inner_product_argument_proof_bn254_curve() {
        const M: usize = 4;
        let gen = BN254Curve::generator();
        let d = 8;
        let mut ipa = IPA::new(d, gen.clone());
        let a = Polynomial::new(&[
            FE_2::from(1_u64),
            FE_2::from(2_u64),
            FE_2::from(3_u64),
            FE_2::from(4_u64),
            FE_2::from(5_u64),
            FE_2::from(6_u64),
            FE_2::from(7_u64),
            FE_2::from(8_u64),
        ]);
        let coef_a = a.clone().coefficients;
        let r = generate_random_element();
        let _p = ipa
            .commit(&a, r.clone(), neutral_element_bn254_curve())
            .unwrap();
        let mut rng = rand::thread_rng();
        let vector: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        let _u = build_group_challenge::<Fr_2, M, BN254Curve>(vector, gen);
        let k = (f64::from(ipa.d).log2()) as usize;
        let mut u: Vec<FE_2> = vec![FE_2::zero(); k];
        for item in u.iter_mut().take(k) {
            let mut rng = rand::thread_rng();
            let vector: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
            *item = build_field_challenge(vector);
        }
        let x = FE_2::from(3_u64);
        let b = powers_of(&x.clone(), ipa.d.try_into().unwrap());
        let v = inner_product_field(&coef_a, &b).unwrap();
        let proof = ipa
            .open(&coef_a, &b, &u, &_u, neutral_element_bn254_curve())
            .unwrap();
        let verif = ipa
            .verify_open(
                &x,
                &v,
                &_p,
                &proof,
                &r,
                &u,
                &_u,
                neutral_element_bn254_curve(),
            )
            .unwrap();
        assert!(verif);
    }
}
