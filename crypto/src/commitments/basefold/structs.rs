use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
};

pub struct BaseFoldProverParams<F: IsField> {
    log_rate: usize,
    table_w_weights: Vec<Vec<(FieldElement<F>, FieldElement<F>)>>,
    table: Vec<Vec<FieldElement<F>>>,
    num_verifier_queries: usize,
    num_vars: usize,
    num_rounds: usize,
    rs_basecode: bool,
}

pub struct BaseFoldVerifierParams<F: IsField> {
    num_vars: usize,
    log_rate: usize,
    num_verifier_queries: usize,
    num_rounds: usize,
    table_w_weights: Vec<Vec<(FieldElement<F>, FieldElement<F>)>>,
    rs_basecode: bool,
}

pub struct BasefoldCommitment<F: IsField> {
    evaluation: Polynomial<F>,
    merkle_tree: Vec<Vec<FieldElement<F>>>,
    bh_evals: Polynomial<F>,
}
