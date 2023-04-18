use sha3::Digest;
use sha3::Sha3_256;

pub fn expand_message(msg: &[u8], dst: &[u8], len_in_bytes: u64) -> Result<Vec<u8>, String> {
    let mut b: Vec<Vec<u8>> = Vec::new();

    let ell = (len_in_bytes as f64 / (256_f64 / 8_f64)).ceil() as u64;
    if ell > 255 {
        return Err("Abort".to_string());
    }

    let dst_prime: Vec<u8> = i2osp(dst.len() as u64, 1)
        .iter()
        .zip(dst)
        .map(|(a, b)| a | b)
        .collect();
    let z_pad = i2osp(0, 64);
    let l_i_b_str = i2osp(len_in_bytes, 2);

    let b_0_mid: Vec<u8> = z_pad
        .iter()
        .zip(msg)
        .zip(l_i_b_str)
        .zip(i2osp(0, 1))
        .zip(dst_prime.clone())
        .map(|((((a, b), c), d), e)| a | b | c | d | e)
        .collect();

    let mut hasher = Sha3_256::new();
    hasher.update(b_0_mid);
    b[0] = hasher.finalize().to_vec();

    let b_1_mid: Vec<u8> = b[0]
        .iter()
        .zip(i2osp(1, 1))
        .zip(dst_prime.clone())
        .map(|((a, b), c)| a | b | c)
        .collect();

    let mut hasher = Sha3_256::new();
    hasher.update(b_1_mid);
    b[1] = hasher.finalize().to_vec();

    for i in 2..ell {
        let b_i_mid: Vec<u8> = strxor(&b[0], &b[(i - 1) as usize])
            .iter()
            .zip(i2osp(i, 1))
            .zip(dst_prime.clone())
            .map(|((a, b), c)| a | b | c)
            .collect();
        let mut hasher = Sha3_256::new();
        hasher.update(b_i_mid);
        let b_i = hasher.finalize().to_vec();
        b[i as usize] = b_i;
    }
    let pseudo_random_bytes = b
        .into_iter()
        .reduce(|acc, b| stror(&acc[..], &b[..]))
        .unwrap()
        .to_vec();

    Ok(pseudo_random_bytes[0..len_in_bytes as usize].to_vec())
}

pub fn i2osp(x: u64, length: u64) -> Vec<u8> {
    let mut x_aux = x;
    let mut digits = Vec::new();
    while x_aux != 0 {
        digits.push((x_aux % 256) as u8);
        x_aux /= 256;
    }
    digits.resize(digits.len() + (length - digits.len() as u64) as usize, 0);
    digits.reverse();
    digits
}

pub fn os2ip(x: &[u8]) -> u64 {
    let mut aux_x = x.to_vec();
    let x_len = aux_x.len();
    aux_x.reverse();
    let mut i = 0_u64;
    for (j, item) in aux_x.iter().enumerate().take(x_len) {
        i += *item as u64 * 256_u64.pow(j as u32);
    }
    i
}

pub fn strxor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(a, b)| a ^ b).collect()
}

pub fn stror(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(a, b)| a | b).collect()
}
