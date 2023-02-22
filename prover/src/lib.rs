pub mod fri;
pub mod air;

pub struct StarkProof {
    // TODO: fill this when we know what a proof entails
}

fn prove() -> StarkProof {
    // * Generate trace polynomials using Winterfell
    // * Generate composition polynomials using Winterfell
    // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
    // * Commit to both polynomials using a Merkle Tree
    // * Do FRI on the composition polynomials
    // * Sample q_1, ..., q_m using Fiat-Shamir
    // * For every q_i, do FRI decommitment
    // * For every trace polynomial t_i, provide the evaluations on every q_i, q_i * g, q_i * g^2

    StarkProof{}
}

fn verify() {}
