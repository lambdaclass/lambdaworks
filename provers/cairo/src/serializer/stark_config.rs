use stark_platinum_prover::Felt252;

#[derive(Debug, Clone)]
pub struct StarkConfig {
    pub traces: TracesConfig,
    pub composition: TableCommitmentConfig,
    pub fri: FriConfig,
    pub proof_of_work: ProofOfWorkConfig,
    pub log_trace_domain_size: Felt252,
    pub n_queries: Felt252,
    pub log_n_cosets: Felt252,
    pub n_verifier_friendly_commitment_layers: Felt252,
}
#[derive(Debug, Clone)]
pub struct TracesConfig {
    pub original: TableCommitmentConfig,
    pub interaction: TableCommitmentConfig,
}
#[derive(Debug, Clone)]
pub struct TableCommitmentConfig {
    pub n_columns: Felt252,
    pub vector: VectorCommitmentConfig,
}
#[derive(Debug, Clone)]
pub struct VectorCommitmentConfig {
    pub height: Felt252,
    pub n_verifier_friendly_commitment_layers: Felt252,
}
#[derive(Debug, Clone)]
pub struct FriConfig {
    pub log_input_size: Felt252,
    pub n_layers: Felt252,
    pub inner_layers: Vec<TableCommitmentConfig>,
    pub fri_step_sizes: Vec<Felt252>,
    pub log_last_layer_degree_bound: Felt252,
}
#[derive(Debug, Clone)]
pub struct ProofOfWorkConfig {
    pub n_bits: Felt252,
}
