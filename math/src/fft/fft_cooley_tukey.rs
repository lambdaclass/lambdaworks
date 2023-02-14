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
    use crate::{field::fields::u64_prime_field::U64FieldElement, polynomial::Polynomial};

    use super::*;
    const MODULUS: u64 = 13;
    type FE = U64FieldElement<MODULUS>;

    /// test case generated with <https://www.nayuki.io/page/number-theoretic-transform-integer-dft>
    #[test]
    fn test_cooley_tukey() {
        let poly = Polynomial::new([6, 0, 10, 7].map(FE::from).to_vec());
        let omega = FE::from(8);

        let twiddles_iter = (0..poly.coefficients().len() as u64).map(|i| omega.pow(i));

        let expected: Vec<FE> = twiddles_iter.map(|x| poly.evaluate(x)).collect();
        let result = cooley_tukey(poly.coefficients().to_owned(), omega, MODULUS);

        assert_eq!(result, expected);
    }
}
