use stark_platinum_prover::Felt252;

#[derive(Debug, Clone)]
pub struct CairoPublicInput {
    pub log_n_steps: Felt252,
    pub range_check_min: Felt252,
    pub range_check_max: Felt252,
    pub layout: Felt252,
    pub dynamic_params: Vec<Felt252>,
    pub n_segments: Felt252,
    pub segments: Vec<SegmentInfo>,
    pub padding_addr: Felt252,
    pub padding_value: Felt252,
    pub main_page_len: Felt252,
    pub main_page: Vec<PubilcMemoryCell>,
    pub n_continuous_pages: Felt252,
    pub continuous_page_headers: Vec<Felt252>,
}

#[derive(Debug, Clone)]
pub struct PubilcMemoryCell {
    pub address: Felt252,
    pub value: Felt252,
}

#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub begin_addr: Felt252,
    pub stop_ptr: Felt252,
}
