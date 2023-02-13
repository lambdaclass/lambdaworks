use crate::field::{element::FieldElement, traits::IsField};

pub fn cooley_tukey<F: IsField>(
    coeffs: Vec<FieldElement<F>>,
    omega: FieldElement<F>,
    _modulus: u64,
) -> Vec<FieldElement<F>> {
    let n = coeffs.len();
    assert!(n.is_power_of_two(), "n should be power of two");
    if n == 1 {
        return coeffs;
    }
    let coeffs_even: Vec<FieldElement<F>> = coeffs.iter().step_by(2).cloned().collect();
    let coeffs_odd: Vec<FieldElement<F>> = coeffs.iter().skip(1).step_by(2).cloned().collect();

    let (y_even, y_odd) = (
        cooley_tukey(coeffs_even, omega.clone(), _modulus),
        cooley_tukey(coeffs_odd, omega.clone(), _modulus),
    );
    let mut y = vec![FieldElement::one(); n];
    for i in 0..n / 2 {
        let a = y_even[i].clone();
        let b = omega.pow(i) * y_odd[i].clone();
        y[i] = a.clone() + b.clone();
        y[i + n / 2] = a - b;
    }
    y
}

#[cfg(test)]
mod test {
    use crate::field::fields::u64_prime_field::U64FieldElement;

    use super::*;
    const MODULUS: u64 = 13;
    type FE = U64FieldElement<MODULUS>;

    /// test case generated with <https://www.nayuki.io/page/number-theoretic-transform-integer-dft>
    #[test]
    fn test_cooley_tukey() {
        let coeffs = vec![FE::new(6), FE::new(0), FE::new(10), FE::new(7)];
        let omega = FE::from(8);

        let result = cooley_tukey(coeffs, omega, MODULUS);
        let expected = vec![FE::new(10), FE::new(5), FE::new(9), FE::new(0)];
        assert_eq!(result, expected);
    }
}
