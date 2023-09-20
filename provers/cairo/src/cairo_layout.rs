#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CairoLayout {
    Plain,
    Small,
    Dex,
    Recursive,
    Starknet,
    StarknetWithKeccak,
    RecursiveLargeOutput,
    AllCairo,
    AllSolidity,
    Dynamic,
}

impl CairoLayout {
    pub fn as_str(&self) -> &'static str {
        match self {
            CairoLayout::Plain => "plain",
            CairoLayout::Small => "small",
            CairoLayout::Dex => "dex",
            CairoLayout::Recursive => "recursive",
            CairoLayout::Starknet => "starknet",
            CairoLayout::StarknetWithKeccak => "starknet_with_keccak",
            CairoLayout::RecursiveLargeOutput => "recursive_large_output",
            CairoLayout::AllCairo => "all_cairo",
            CairoLayout::AllSolidity => "all_solidity",
            CairoLayout::Dynamic => "dynamic",
        }
    }
}
