use crate::{
    fft::helpers::pad_to_next_power_of_two,
    field::{
        element::FieldElement,
        traits::{IsField, IsPrimeField, IsTwoAdicField, RootsConfig},
    },
    polynomial::Polynomial,
};

use super::{
    bit_reversing::in_place_bit_reverse_permute, errors::FFTError,
    fft_iterative::in_place_nr_2radix_fft, helpers::log2,
};

pub fn fast_multiply<F: IsField + IsTwoAdicField + IsPrimeField>(
    multiplicand_digits: &mut Vec<FieldElement<F>>,
    multiplier_digits: &mut Vec<FieldElement<F>>,
) -> Result<FieldElement<F>, FFTError>
where
    <F as IsField>::BaseType: From<<F as IsPrimeField>::RepresentativeType>,
{
    pad_to_next_power_of_two(multiplicand_digits);
    pad_to_next_power_of_two(multiplier_digits);

    let multiplicand_evaluation_result = Polynomial::evaluate_offset_fft(
        &Polynomial::new(multiplicand_digits),
        &FieldElement::one(),
        2,
    )?;
    let multiplier_evaluation_result = Polynomial::evaluate_offset_fft(
        &Polynomial::new(multiplier_digits),
        &FieldElement::one(),
        2,
    )?;

    // Multiply the evaluations of both polynomials and then apply the inverse FFT
    // to get the original result of the multiplication
    // This works since:
    // FFT(c) = FFT(a).FFT(b)
    // IFFT(FFT(c)) = IFFT(FFT(a).FFT(b))
    // c = a.b
    let evaluation_points_multplication_result = multiplicand_evaluation_result
        .iter()
        .zip(multiplier_evaluation_result.iter())
        .map(|(_a, _b)| _a * _b)
        .collect::<Vec<FieldElement<F>>>();
    let coeffs_multiplication_result =
        Polynomial::interpolate_fft(&evaluation_points_multplication_result).unwrap();

    // Evaluate the polynomial to get the original result from the new polynomial
    let h = Polynomial::new(&coeffs_multiplication_result.coefficients);
    let result = h.evaluate(&FieldElement::from(10));

    Ok(result)
}

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F` in a coset. Usually used for
/// fast polynomial evaluation.
pub fn fft_with_blowup<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
    blowup_factor: usize,
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let domain_size = coeffs.len() * blowup_factor;
    let order = log2(domain_size)?;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverse)?;

    let mut results = coeffs.to_vec();
    results.resize(domain_size, FieldElement::zero());

    in_place_nr_2radix_fft(&mut results, &twiddles);
    in_place_bit_reverse_permute(&mut results);

    Ok(results)
}

/// Executes Fast Fourier Transform over elements of a two-adic finite field `F`. Usually used for
/// fast polynomial evaluation.
pub fn fft<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let order = log2(coeffs.len())?;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverse)?;

    let mut results = coeffs.to_vec();
    in_place_nr_2radix_fft(&mut results[..], &twiddles[..]);
    in_place_bit_reverse_permute(&mut results);

    Ok(results)
}

/// Executes the inverse Fast Fourier Transform over elements of a two-adic finite field `F`.
/// Usually used for fast polynomial evaluation.
pub fn inverse_fft<F: IsTwoAdicField>(
    coeffs: &[FieldElement<F>],
) -> Result<Vec<FieldElement<F>>, FFTError> {
    let order = log2(coeffs.len())?;
    let twiddles = F::get_twiddles(order, RootsConfig::BitReverseInversed)?;

    let mut results = coeffs.to_vec();
    in_place_nr_2radix_fft(&mut results[..], &twiddles[..]);
    in_place_bit_reverse_permute(&mut results);

    for elem in &mut results {
        *elem = elem.clone() / FieldElement::from(coeffs.len() as u64); // required for inverting the DFT matrix.
    }

    Ok(results)
}

#[cfg(test)]
mod test {
    use crate::field::{
        element::FieldElement,
        fields::fft_friendly::u256_two_adic_prime_field::U256MontgomeryTwoAdicPrimeField,
    };

    use super::fast_multiply;

    type F = U256MontgomeryTwoAdicPrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn fft_multiplication() {
        let mut multplicand_digits = vec![FE::from(1), FE::from(0), FE::from(2)];
        let mut multiplier_digits = multplicand_digits.clone();
        let a = fast_multiply(&mut multplicand_digits, &mut multiplier_digits).unwrap();
        assert_eq!(a, FE::from(201) * FE::from(201))
    }
}
