use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::IsField
    },
    polynomial::Polynomial,
};

use rand::random;
use rand::prelude::Distribution;
use rand::distributions::Standard;

pub struct ShamirSecretSharing<F: IsField> {
    secret: FieldElement<F>,
    n: usize,
    k: usize,
}

pub struct Shares<F: IsField> {
    pub x: Vec<FieldElement<F>>,
    pub y: Vec<FieldElement<F>>,
}

impl<F: IsField> ShamirSecretSharing<F> {
    pub fn calculate_polynomial(&self) -> Polynomial<FieldElement<F>>
    where
        Standard: Distribution<<F as IsField>::BaseType>,
    {
        let mut coefficients = Vec::new();
        coefficients.push(self.secret.clone());
        for _ in 0..self.k - 1 {
            coefficients.push(FieldElement::<F>::new(random()));
        }

        let polynomial = Polynomial::new(coefficients.as_slice());
        polynomial
    }

    pub fn generating_shares(&self, polynomial: Polynomial<FieldElement<F>>) -> Shares<F>
    where
        Standard: Distribution<<F as IsField>::BaseType>,
    {
        let mut shares_x: Vec<FieldElement<F>> = Vec::new();
        let mut shares_y: Vec<FieldElement<F>> = Vec::new();

        for _ in 0..self.n {
            let x = FieldElement::<F>::new(random());
            let y = polynomial.evaluate(&x);
            shares_x.push(x);
            shares_y.push(y);
        }

        Shares {
            x: shares_x,
            y: shares_y,
        }
    }

    pub fn reconstructing(
        &self,
        x: Vec<FieldElement<F>>,
        y: Vec<FieldElement<F>>,
    ) -> Polynomial<FieldElement<F>> {
        let poly_reconstructed = Polynomial::interpolate(&x, &y).unwrap();
        poly_reconstructed
    }

    pub fn recover(&self, polynomial: &Polynomial<FieldElement<F>>) -> FieldElement<F> {
        polynomial.coefficients()[0].clone()
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {

    use lambdaworks_math::field::element::FieldElement;
    use crate::shamir_secret_sharing::ShamirSecretSharing;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    #[test]
    fn shamir_secret_sharing_works() {

    const ORDER: u64 = 1613;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    let sss = ShamirSecretSharing {
        secret: FE::new(1234),
        n: 6,
        k: 3,
    };

    let polynomial = sss.calculate_polynomial();
    let shares = sss.generating_shares(polynomial.clone());
    let shares_to_use_x = vec![shares.x[1], shares.x[3], shares.x[4]];
    let shares_to_use_y = vec![shares.y[1], shares.y[3], shares.y[4]];
    let poly_2 = sss.reconstructing(shares_to_use_x, shares_to_use_y);
    let secret_recovered = sss.recover(&poly_2);
    assert_eq!(polynomial, poly_2);
    assert_eq!(sss.secret, secret_recovered);
    }    
}
