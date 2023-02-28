use super::{Polynomial, FE};

fn fold_polynomial(poly: &Polynomial<FE>, beta: &FE) -> Polynomial<FE> {
    let coef = poly.coefficients();
    let even_coef: Vec<FE> = coef.iter().step_by(2).cloned().collect();

    // odd coeficients of poly are multiplied by beta
    let odd_coef_mul_beta: Vec<FE> = coef
        .iter()
        .skip(1)
        .step_by(2)
        .map(|v| (v.clone()) * beta)
        .collect();

    let (even_poly, odd_poly) = Polynomial::pad_with_zero_coefficients(
        &Polynomial::new(&even_coef),
        &Polynomial::new(&odd_coef_mul_beta),
    );
    even_poly + odd_poly
}

fn next_domain(input: &[FE]) -> Vec<FE> {
    let length = input.len() / 2;
    let mut ret = Vec::with_capacity(length);
    for v in input.iter().take(length) {
        ret.push(v * v)
    }
    ret
}

/// Returns:
/// * new polynomoial folded with FRI protocol
/// * new domain
/// * evaluations of the polynomial
pub fn next_fri_layer(
    poly: &Polynomial<FE>,
    domain: &[FE],
    beta: &FE,
) -> (Polynomial<FE>, Vec<FE>, Vec<FE>) {
    let ret_poly = fold_polynomial(poly, beta);
    let ret_next_domain = next_domain(domain);
    println!("NEXT DOMAIN: {:?}", ret_next_domain[2]);
    let ret_evaluation = ret_poly.evaluate_slice(&ret_next_domain);
    println!("NEXT EVALUATION: {:?}", ret_evaluation[2]);
    (ret_poly, ret_next_domain, ret_evaluation)
}

#[cfg(test)]
mod tests {
    use super::{fold_polynomial, next_domain, next_fri_layer, FE};
    use lambdaworks_math::polynomial::Polynomial;
    use lambdaworks_math::unsigned_integer::element::U384;
    use parameterized_test;

    parameterized_test::create! { fold, (p0, beta, expected_p1), {
        let p1 = fold_polynomial(&p0, &beta);
        assert_eq!(p1, expected_p1);
    }}
    fold! {
        first_fold: (poly(&[3, 1, 2, 7, 3, 5,]),    // p0
                     FE::new(U384::from_u64(4)),    // beta
                     poly(&[7, 30, 23,])),          // p1
        second_fold: (poly(&[7, 30, 23,]),          // same p1
                      FE::new(U384::from_u64(3)),   // gamma
                      poly(&[97, 23,])),            // p2
        third_fold: (poly(&[97, 23,]),              // same p2
                     FE::new(U384::from_u64(2)),    // delta
                     poly(&[143,])),                // p3
    }

    parameterized_test::create! { next_domain, (input_domain, expected_ret_domain), {
        let ret_next_domain = next_domain(&input_domain);
        assert_eq!(ret_next_domain, expected_ret_domain);
    }}
    next_domain! {
        first: (domain(&[5, 7, 13, 20, 1, 1, 1, 1,]),     // Input domain
                domain(&[25, 49, 169, 107,])),            // Expected next domain
        second: (domain(&[25, 49, 169, 107,]),            // Same as previous result
                 domain(&[39, 57])),                      // Expected next domain
        third: (domain(&[39, 57]),                        // Third iteration
                domain(&[56])),                           // Expected final result
    }

    parameterized_test::create! { next_fri_layer, ((p0, beta, input_domain), (expected_p1, expected_domain, expected_evaluation)), {
        let (p1, ret_next_domain, ret_evaluation) = next_fri_layer(&p0, &input_domain, &beta);
        assert_eq!(p1, expected_p1);
        assert_eq!(ret_next_domain, expected_domain);
        assert_eq!(ret_evaluation, expected_evaluation);
    }}
    next_fri_layer! {
        basic_case: ((poly(&[3, 1, 2, 7, 3, 5,]),               // Input poly
                      FE::new(U384::from_u64(3)),               // Beta
                      domain(&[5, 7, 13, 20, 1, 1, 1, 1])),     // Input Domain
                     (poly(&[7, 30, 23,]),                      // Expected poly
                      domain(&[25, 49, 169, 107,]),             // Expected return domain
                      domain(&[189, 151, 93, 207,]))),          // Expected return evaluation
    }

    // Helper functions for shortest declarations
    fn poly(coefficients: &[u64]) -> Polynomial<FE> {
        Polynomial::new(&domain(coefficients))
    }

    fn domain(coefficients: &[u64]) -> Vec<FE> {
        coefficients
            .iter()
            .map(|f| FE::new(U384::from_u64(*f)))
            .collect::<Vec<FE>>()
    }
}
