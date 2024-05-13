use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use lambdaworks_math::field::element::FieldElement;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use num::integer::gcd;


pub fn linear_combination_u64(u: &[u64], v: &[FieldElement<Mersenne31Field>]) -> FieldElement<Mersenne31Field> {
    assert_eq!(u.len(), v.len(), "The lengths of u and v must be the same.");

    let mut result = FieldElement::<Mersenne31Field>::zero();
    
    for (ui, vi) in u.iter().zip(v.iter()) {
        
        result = result + FieldElement::<Mersenne31Field>::from(*ui) * vi;
    }

    result
}

pub fn rotate_right<const N: usize>(input: [u64; N], offset: usize) -> [u64; N] {
    let mut output = [0u64; N];
    let offset = offset % N; 
    for (i, item) in input.iter().enumerate() {
        output[(i + offset) % N] = *item;
    }
    output
}

pub fn add_round_constants(state: &mut Vec<FieldElement<Mersenne31Field>> , round_constants: Vec<FieldElement<Mersenne31Field>>) {
    for (s, rc) in state.iter_mut().zip(round_constants.iter()) {
        *s = *s + rc; 
    }
}

pub fn sbox(state: &mut Vec<FieldElement<Mersenne31Field>>, alpha: u64) {
    for elem in state.iter_mut() {
        *elem = elem.pow(alpha);  
    }
}

pub fn sbox_inv(state: &mut Vec<FieldElement<Mersenne31Field>>, alpha_inv: u64) {
    for elem in state.iter_mut() {
        *elem = elem.pow(alpha_inv);  
    }
}

pub fn shake256_hash(input: &[u8], output_len: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(input);
    let mut res = vec![0; output_len];
    hasher.finalize_xof().read(&mut res);
    res
}


pub fn get_alphas(p: u64) -> (u64, u64) {
    let mut alpha = 0;
    let mut alphainv = 0;

    fn extended_gcd(a: u64, b: u64) -> (u64, i64, i64) {
        if a == 0 {
            (b, 0, 1)
        } else {
            let (g, x, y) = extended_gcd(b % a, a);
            (g, y - (b / a) as i64 * x, x)
        }
    }

    for a in 3..p {
        if gcd(a, p - 1) == 1 {
            alpha = a;
            let (g, x, _) = extended_gcd(a, p - 1);
            if g == 1 {
                alphainv = x.rem_euclid((p - 1) as i64) as u64; 
                break;
            }
        }
    }

    (alpha, alphainv)
}


pub fn binomial(mut n: UnsignedInteger<4>, k: UnsignedInteger<4>) -> UnsignedInteger<4> {
    if k > n {
        return UnsignedInteger::from_u64(0);
    }
    if k > n.clone() - k.clone() {
        return binomial(n.clone(), n - k);
    }
    let mut r = UnsignedInteger::from_u64(1);
    let mut d = UnsignedInteger::from_u64(1);
    while d <= k {
        r = multiply_and_divide(r, n.clone(), d.clone());
        n = n - UnsignedInteger::from_u64(1);
        d = d + UnsignedInteger::from_u64(1);
    }
    r
}

fn multiply_and_divide(a: UnsignedInteger<4>, b: UnsignedInteger<4>, c: UnsignedInteger<4>) -> UnsignedInteger<4> {
    let a_mul_b = a * b; 
    let (quotient, _remainder) = a_mul_b.div_rem(&c);
    quotient
}


    

    

    
    
    
    
    
    
