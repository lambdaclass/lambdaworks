use lambdaworks_math::{
    circle::point::CirclePoint,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

pub fn evaluate_vanishing_poly(
    log_2_size: u32,
    point: CirclePoint<Mersenne31Field>,
) -> FieldElement<Mersenne31Field> {
    let mut x = point.x;
    for _ in 1..log_2_size {
        x = x.square().double() - FieldElement::one();
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::circle::cosets::Coset;

    type FE = FieldElement<Mersenne31Field>;

    #[test]
    fn vanishing_poly_vanishes_in_coset() {
        let log_2_size = 3;
        let coset = Coset::new_standard(log_2_size);
        let points = Coset::get_coset_points(&coset);
        for point in points {
            assert_eq!(evaluate_vanishing_poly(log_2_size, point), FE::zero());
        }
    }
    #[test]
    fn vanishing_poly_doesnt_vanishe_outside_coset() {
        let log_2_size = 3;
        let coset = Coset::new_standard(log_2_size + 1);
        let points = Coset::get_coset_points(&coset);
        for point in points {
            assert_ne!(evaluate_vanishing_poly(log_2_size, point), FE::zero());
        }
    }
}
