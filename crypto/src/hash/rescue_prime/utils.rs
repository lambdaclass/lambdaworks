use super::Fp;
use alloc::vec::Vec;

pub fn bytes_to_field_elements(input: &[u8]) -> Vec<Fp> {
    input
        .chunks(7)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            if chunk.len() < 7 {
                buf[chunk.len()] = 1;
            }
            let value = u64::from_le_bytes(buf);
            Fp::from(value)
        })
        .collect()
}

pub fn ntt(input: &[Fp], omega: Fp) -> Vec<Fp> {
    (0..input.len())
        .map(|i| {
            input.iter().enumerate().fold(Fp::zero(), |acc, (j, val)| {
                acc + *val * omega.pow((i * j) as u64)
            })
        })
        .collect()
}

pub fn intt(input: &[Fp], omega_inv: Fp) -> Vec<Fp> {
    let inv_n = Fp::from(input.len() as u64).inv().unwrap();
    ntt(input, omega_inv)
        .into_iter()
        .map(|val| val * inv_n)
        .collect()
}

pub fn karatsuba(lhs: &[Fp], rhs: &[Fp]) -> Vec<Fp> {
    let n = lhs.len();
    if n <= 32 {
        let mut result = vec![Fp::zero(); 2 * n - 1];
        lhs.iter().enumerate().for_each(|(i, &lhs_val)| {
            rhs.iter().enumerate().for_each(|(j, &rhs_val)| {
                result[i + j] += lhs_val * rhs_val;
            });
        });
        return result;
    }

    let half = n / 2;
    let (lhs_low, lhs_high) = lhs.split_at(half);
    let (rhs_low, rhs_high) = rhs.split_at(half);

    let z0 = karatsuba(lhs_low, rhs_low);
    let z2 = karatsuba(lhs_high, rhs_high);

    let lhs_sum: Vec<Fp> = lhs_low.iter().zip(lhs_high).map(|(a, b)| *a + *b).collect();
    let rhs_sum: Vec<Fp> = rhs_low.iter().zip(rhs_high).map(|(a, b)| *a + *b).collect();

    let z1 = karatsuba(&lhs_sum, &rhs_sum);

    let mut result = vec![Fp::zero(); 2 * n - 1];

    z0.iter().enumerate().for_each(|(i, &val)| result[i] = val);
    z2.iter()
        .enumerate()
        .for_each(|(i, &val)| result[i + 2 * half] = val);

    z1.iter().enumerate().for_each(|(i, &val)| {
        result[i + half] += val
            - z0.get(i).cloned().unwrap_or(Fp::zero())
            - z2.get(i).cloned().unwrap_or(Fp::zero());
    });

    result
}
