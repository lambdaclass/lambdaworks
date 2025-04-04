use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
            point::ShortWeierstrassProjectivePoint,
        },
        traits::FromAffine,
    },
    field::element::FieldElement,
    unsigned_integer::element::U384,
};

use crate::chinese_remainder_theorem::*;

/// Errors that can occur during Pohlig-Hellman algorithm execution
#[derive(Debug)]
pub enum PohligHellmanError {
    /// Error when no discrete logarithm solution could be found
    DiscreteLogNotFound,
    /// Error from Chinese Remainder Theorem calculation
    ChineseRemainderTheoremError(ChineseRemainderTheoremError),
}

/// Represents a group suitable for the Pohlig-Hellman algorithm
#[derive(Clone, Debug)]
pub struct PohligHellmanGroup {
    pub order: u64,
    pub order_factors: [u64; 3],
    pub generator: ShortWeierstrassProjectivePoint<BLS12381Curve>,
}

impl PohligHellmanGroup {
    /// Creates a new subgroup of the BLS12-381 curve with a smooth order
    /// that is suitable for the Pohlig-Hellman algorithm.
    pub fn new() -> Self {
        // Big subgroup of the BLS12-381 Elliptic Curve.
        // Its order's factorization is: 11 * 10177 * 859267 * 52437899 * 52435875175126190479447740508185965837690552500527637822603658699938581184513.
        // We'll take the first three factors to form the subgroup we are going to attack.
        let group_order = U384::from_hex_unchecked(
            "1FB322654A7CEF70462F7D205CF17F1D6B52ECA5FE8D9BBD809536AAD8A973FFF0AAAAAA5555AAAB",
        );

        // Generator of the group of order `big_group_order`.
        let group_generator = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(
            FieldElement::<BLS12381PrimeField>::from_hex_unchecked("04"),
            FieldElement::<BLS12381PrimeField>::from_hex_unchecked("0a989badd40d6212b33cffc3f3763e9bc760f988c9926b26da9dd85e928483446346b8ed00e1de5d5ea93e354abe706c"),
        ).unwrap();

        // We take the first three factors of the big group order, forming the order of a subgroup.
        // 96192362849 = 11 * 10177 * 859267.
        let subgroup_order = 96192362849u64;

        let subgroup_order_factors = [11u64, 10177u64, 859267u64];

        // We construct the generator 'h' of the subgroup using:
        // h = g^{n/s}, where 'g' is the big group generator, 'n' is its order and 's' the subgroup order.
        let quotient = group_order.div_rem(&U384::from(subgroup_order)).0;
        let subgroup_generator = group_generator.operate_with_self(quotient);

        Self {
            order: subgroup_order,
            order_factors: subgroup_order_factors,
            generator: subgroup_generator,
        }
    }

    /// Performs the Pohlig-Hellman attack to find the discrete logarithm
    ///
    /// Given a point q = g^x, finds the value x using the Pohlig-Hellman algorithm.
    pub fn pohlig_hellman_attack(
        self,
        q: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
    ) -> Result<usize, PohligHellmanError> {
        // In this vector we'll collect all the equations that we are going to use to find the exponent x using the Chinese Remainder Theorem.
        // The elements of `equations` will be of the form (x', n), which represents that x satisfies x ≡ x' mod n.
        let mut equations = Vec::new();

        // For each factor n in the factorization of the order, we search for x' such that
        // x ≡ x' mod n, using the Baby Step Giant Step algorithm.
        for factor in self.order_factors {
            let cofactor = self.order / factor;

            // We look for x' such that {h'}^{x'} = q'
            let h_prime = self.generator.operate_with_self(cofactor);
            let q_prime = q.operate_with_self(cofactor);

            // We use Baby Step Giant Step to find the discrete logarithm in the subgroup of order `factor`.
            if let Some(x_prime) = Self::baby_step_giant_step(&h_prime, &q_prime, factor) {
                // We add the congruence x ≡ x' mod factor
                equations.push((x_prime as i128, factor as i128));
            } else {
                // If we couldn't find the discrete logarithm in the subgroup of order `factor`, it
                // means that there is no solution to the discrete logarithm equation proposed.
                // That is, there is no 'k' such that h^k = q.
                return Err(PohligHellmanError::DiscreteLogNotFound);
            }
        }

        // We combine the equations using the Chinese Remainder Theorem.
        let x = chinese_remainder_theorem(&equations)?;
        Ok(x as usize)
    }

    /// Implementation of the Baby-Step Giant-Step algorithm
    ///
    /// Finds the discrete logarithm x such that h^x = q in a cyclic group,
    /// where 0 ≤ x < n.
    pub fn baby_step_giant_step(
        h: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
        q: &ShortWeierstrassProjectivePoint<BLS12381Curve>,
        n: u64,
    ) -> Option<u64> {
        // Compute optimal step size m ≈ sqrt(n)
        let m: u64 = (n as f64).sqrt().ceil() as u64 + 1;

        // We construct the list `baby_list`= [h^0, h^1, ..., h^{m-1}].
        let mut baby_list = Vec::with_capacity(m as usize);
        for j in 0..m {
            baby_list.push(h.operate_with_self(j));
        }

        // We want to compare the lists [h^0, h^1, ..., h^{m-1}] and [q * h^{-m}, q * h^{-2m}, q * h^{-3m}, ... , q * h^{-m*m}]
        let hm = h.operate_with_self(m);
        let minus_hm = hm.neg();

        // y = q initially
        let mut y = q.clone();

        for i in 0..m {
            // We check if y is in the baby_list
            for j in 0..m {
                if baby_list[j as usize] == y {
                    let x = i * m + j;
                    return Some(x);
                }
            }

            // y = y * h^{-m}
            y = y.operate_with(&minus_hm);
        }

        // If there isn't any coincidence between lists, then there is no result x such that h^x = q.
        None
    }
}

// Convert from ChineseRemainderTheoremError to PohligHellmanError
impl From<ChineseRemainderTheoremError> for PohligHellmanError {
    fn from(error: ChineseRemainderTheoremError) -> Self {
        PohligHellmanError::ChineseRemainderTheoremError(error)
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    use lambdaworks_math::{elliptic_curve::traits::FromAffine, unsigned_integer::element::U384};

    use super::*;

    #[test]
    fn test_mod_inverse() {
        // Test case 1: Simple case
        assert_eq!(mod_inverse(3, 7), Some(5));

        // Test case 2: Larger numbers
        assert_eq!(mod_inverse(17, 3120), Some(2753));

        // Test case 3: Non-invertible case
        assert_eq!(mod_inverse(2, 4), None);

        // Test case 4: Identity case
        assert_eq!(mod_inverse(1, 5), Some(1));
    }

    #[test]
    fn test_chinese_remainder_theorem() {
        // Test case 1: Simple case
        let equations = [(2, 3), (3, 5), (2, 7)];
        assert_eq!(chinese_remainder_theorem(&equations).unwrap(), 23);

        // Test case 2: Our Pohlig-Hellman case
        let equations = [(1, 4), (2, 27)];
        assert_eq!(chinese_remainder_theorem(&equations).unwrap(), 29);

        // Test case 3: Another common case
        let equations = [(1, 3), (2, 4), (3, 5)];
        assert_eq!(chinese_remainder_theorem(&equations).unwrap(), 58);
    }

    #[test]
    fn new_group() {
        let g = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(
            FieldElement::<BLS12381PrimeField>::from_hex_unchecked("04"),
            FieldElement::<BLS12381PrimeField>::from_hex_unchecked("0a989badd40d6212b33cffc3f3763e9bc760f988c9926b26da9dd85e928483446346b8ed00e1de5d5ea93e354abe706c"),
        ).unwrap();

        let order = U384::from_dec_str("264493130243071837505502650427723022029376893320131317353094445957672484527674560789128274094763").unwrap();

        assert_eq!(
            g.operate_with_self(order),
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::neutral_element()
        );

        // s = 11 * 10177
        let s = U384::from(111947u64);

        // t = r / s
        let (t, _) = order.div_rem(&s);

        // h generator of the subgroup of order s
        let h = g.operate_with_self(t);
        assert_eq!(
            h.operate_with_self(s),
            ShortWeierstrassProjectivePoint::<BLS12381Curve>::neutral_element()
        );
    }

    #[test]
    fn test_pohlig_hellman_attack() {
        let test = PohligHellmanGroup::new();

        let n = 100000000000;
        let q5 = test.generator.operate_with_self(n);
        assert_eq!(
            test.clone().pohlig_hellman_attack(&q5).unwrap(),
            (n % test.order) as usize
        );
    }
}
