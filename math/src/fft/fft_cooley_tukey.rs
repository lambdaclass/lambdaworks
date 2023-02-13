use crate::field::{element::FieldElement, traits::IsField};

pub fn fft<F: IsField>(coefficients: Vec<FieldElement<F>>, modulus: u64) -> Vec<FieldElement<F>> {
    let omega = find_omega(coefficients.len() as u64, modulus);
    cooley_tukey(coefficients, omega, modulus)
}

pub fn cooley_tukey<F: IsField>(
    coeffs: Vec<FieldElement<F>>,
    omega: FieldElement<F>,
    modulus: u64,
) -> Vec<FieldElement<F>> {
    let n = coeffs.len();
    assert!(n.is_power_of_two(), "n should be power of two");
    if n == 1 {
        return coeffs;
    }
    let coeffs_even: Vec<FieldElement<F>> = coeffs.iter().step_by(2).cloned().collect();
    let coeffs_odd: Vec<FieldElement<F>> = coeffs.iter().skip(1).step_by(2).cloned().collect();

    let (y_even, y_odd) = (
        cooley_tukey(coeffs_even, omega.clone(), modulus),
        cooley_tukey(coeffs_odd, omega.clone(), modulus),
    );
    let mut y = vec![FieldElement::one(); n];
    for i in 0..n / 2 {
        let a = y_even[i].clone();
        let b = omega.pow(n * i) * y_odd[i].clone();
        y[i] = a.clone() + b.clone();
        y[i + n / 2] = a - b + FieldElement::from(modulus);
    }
    y
}

fn find_generator<F: IsField>(modulus: u64) -> FieldElement<F> {
    let max_value = modulus - 1;
    let prime_factors = prime_factors_of(max_value as i64);
    for generator in 1..modulus {
        if prime_factors.iter().all(|factor| {
            FieldElement::<F>::one()
                != FieldElement::from(generator).pow(max_value / *factor as u64)
        }) {
            return FieldElement::from(generator);
        }
    }

    panic!("No generator exists under the modulus {modulus}")
}

fn find_omega<F: IsField>(n: u64, modulus: u64) -> FieldElement<F> {
    let k = (modulus - 1) / n;
    let generator: FieldElement<F> = find_generator(modulus);
    generator.pow(k)
}

fn prime_factors_of(mut number: i64) -> Vec<i64> {
    if is_prime(number) {
        return vec![number];
    }

    let mut factors = Vec::new();
    let half = number / 2 + 1;
    let mut pushed = false;

    for factor in 2..half {
        if number == 1 {
            break;
        }
        while 0 == number % factor {
            number /= factor;
            if !pushed {
                factors.push(factor);
                pushed = true;
            }
        }
        pushed = false;
    }

    factors
}

fn is_prime(number: i64) -> bool {
    if number == 2 {
        return true;
    }
    if number == 0 || number == 1 {
        return false;
    }

    let sqrt = (number as f64).sqrt() as i64 + 1;
    for divisor in (3..=sqrt).step_by(2) {
        if 0 == number % divisor {
            return false;
        }
    }
    true
}

// #[cfg(test)]
// mod test {
//     use crate::field::fields::u64_prime_field::U64PrimeField;

//     use super::*;
//     const MODULUS: u64 = 11;
//     type FE = FieldElement<U64PrimeField<MODULUS>>;

//     #[test]
//     fn xx() {
//         let pol = vec![FE::new(6), FE::new(0), FE::new(10), FE::new(7), FE::new(2)];
//         let omega: FieldElement<U64PrimeField<MODULUS>> = find_omega(pol.len() as u64, MODULUS);
//         println!("{:?}", fft(pol, MODULUS));
//         println!("{:?}", omega);
//     }
// }
