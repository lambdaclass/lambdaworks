use lambdaworks_math::{
    circle::point::CirclePoint,
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

// See https://vitalik.eth.limo/general/2024/07/23/circlestarks.html (Section: Quetienting).
// https://github.com/ethereum/research/blob/master/circlestark/line_functions.py#L10
pub fn line(
    eval_point: &CirclePoint<Mersenne31Field>,
    first_vanish_point: &CirclePoint<Mersenne31Field>,
    second_vanish_point: &CirclePoint<Mersenne31Field>,
) -> FieldElement<Mersenne31Field> {
    (first_vanish_point.y - second_vanish_point.y) * eval_point.x
        + (second_vanish_point.x - first_vanish_point.x) * eval_point.y
        + (first_vanish_point.x * second_vanish_point.y
            - first_vanish_point.y * second_vanish_point.x)
}

// See https://vitalik.eth.limo/general/2024/07/23/circlestarks.html (Section: Quetienting).
// https://github.com/ethereum/research/blob/master/circlestark/line_functions.py#L16
// Evaluates the polybomial I at eval_point. I is the polynomial such that I(point_1) = value_1 and
// I(point_2) = value_2.
pub fn interpolant(
    point_1: &CirclePoint<Mersenne31Field>,
    point_2: &CirclePoint<Mersenne31Field>,
    value_1: FieldElement<Mersenne31Field>,
    value_2: FieldElement<Mersenne31Field>,
    eval_point: &CirclePoint<Mersenne31Field>,
) -> FieldElement<Mersenne31Field> {
    let dx = point_2.x - point_1.x;
    let dy = point_2.y - point_1.y;
    // CHECK: can dx^2 + dy^2 = 0 even if dx!=0 and dy!=0 ? (using that they are FE of Mersenne31).
    let invdist = (dx * dx + dy * dy).inv().unwrap();
    let dot = (eval_point.x - point_1.x) * dx + (eval_point.y - point_1.y) * dy;
    value_1 + (value_2 - value_1) * dot * invdist
}
