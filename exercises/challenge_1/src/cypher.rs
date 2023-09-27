use crate::field::ChallengeElement;

const ROUNDS: usize = 2_usize.pow(24);

pub fn evaluate(x: &ChallengeElement, key: &ChallengeElement) -> ChallengeElement {
    (0..ROUNDS).fold(x.clone(), |acc, _| evaluate_round(&acc, key))
}

pub fn evaluate_round(x: &ChallengeElement, key: &ChallengeElement) -> ChallengeElement {
    (x + key).pow(2_u64)
}
