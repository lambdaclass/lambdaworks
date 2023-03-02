use super::element::UnsignedInteger;

pub struct MontgomeryAlgorithms;
impl MontgomeryAlgorithms {
    /// Compute CIOS multiplication of `a` * `b`
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{64}
    /// Notice CIOS stands for Coarsely Integrated Operand Scanning
    /// For more information see section 2.3.2 of Tolga Acar's thesis
    /// https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf
    pub const fn cios<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let mut t = [0_u64; NUM_LIMBS];
        let mut t_extra = [0_u64; 2];
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // C := 0
            let mut c: u128 = 0;

            // for j=0 to N-1
            //    (C,t[j]) := t[j] + a[j]*b[i] + C
            let mut cs: u128;
            let mut j: usize = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + (a.limbs[j] as u128) * (b.limbs[i] as u128) + c;
                c = cs >> 64;
                t[j] = ((cs << 64) >> 64) as u64;
            }

            // (t[N+1],t[N]) := t[N] + C
            cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = ((cs << 64) >> 64) as u64;

            let mut c: u128;

            // m := t[0]*q'[0] mod D
            let m = ((t[NUM_LIMBS - 1] as u128 * *mu as u128) << 64) >> 64;

            // (C,_) := t[0] + m*q[0]
            c = (t[NUM_LIMBS - 1] as u128 + m * (q.limbs[NUM_LIMBS - 1] as u128)) >> 64;

            // for j=1 to N-1
            //    (C,t[j-1]) := t[j] + m*q[j] + C
            let mut j: usize = NUM_LIMBS - 1;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + m * (q.limbs[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = ((cs << 64) >> 64) as u64;
            }

            // (C,t[N-1]) := t[N] + C
            cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = ((cs << 64) >> 64) as u64;

            // t[N] := t[N+1] + C
            t_extra[1] = t_extra[0] + c as u64;
        }
        let mut result = UnsignedInteger { limbs: t };

        let overflow = t_extra[0] > 0;
        // TODO: assuming the integer represented by
        // [t_extra[1], t[0], ..., t[NUM_LIMBS - 1]] is at most
        // 2q in any case.
        if overflow || UnsignedInteger::const_le(q, &result) {
            (result, _) = UnsignedInteger::sub(&result, q);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::unsigned_integer::{element::U384, montgomery::MontgomeryAlgorithms};

    #[test]
    fn montgomery_multiplication_works_0() {
        let x = U384::from_u64(11_u64);
        let y = U384::from_u64(10_u64);
        let m = U384::from_u64(23_u64); //
        let mu: u64 = 3208129404123400281; // negative of the inverse of `m` modulo 2^{64}.
        let c = U384::from_u64(13_u64); // x * y * (r^{-1}) % m, where r = 2^{64 * 6} and r^{-1} mod m = 2.
        assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
    }

    #[test]
    fn montgomery_multiplication_works_1() {
        let x = U384::from("05ed176deb0e80b4deb7718cdaa075165f149c");
        let y = U384::from("5f103b0bd4397d4df560eb559f38353f80eeb6");
        let m = U384::from("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
        let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
        let c = U384::from("8d65cdee621682815d59f465d2641eea8a1274dc"); // x * y * (r^{-1}) % m, where r = 2^{64 * 6}
        assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
    }

    #[test]
    fn montgomery_multiplication_works_3() {
        let x = U384::from("8d65cdee621682815d59f465d2641eea8a1274dc");
        let m = U384::from("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
        let r_mod_m = U384::from("58dfb0e1b3dd5e674bdcde4f42eb5533b8759d33");
        let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
        let c = U384::from("8d65cdee621682815d59f465d2641eea8a1274dc");
        assert_eq!(MontgomeryAlgorithms::cios(&x, &r_mod_m, &m, &mu), c);
    }
}
