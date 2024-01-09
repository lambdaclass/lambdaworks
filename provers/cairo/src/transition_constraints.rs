use crate::Felt252;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::{
    constraints::transition::TransitionConstraint, frame::Frame, table::TableView,
};

#[derive(Clone)]
pub struct BitPrefixFlag0;
impl BitPrefixFlag0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag1;
impl BitPrefixFlag1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        1
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag2;
impl BitPrefixFlag2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        2
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag3;
impl BitPrefixFlag3 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        3
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag4;
impl BitPrefixFlag4 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag4 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        4
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag5;
impl BitPrefixFlag5 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag5 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        5
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag6;
impl BitPrefixFlag6 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag6 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        6
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag7;
impl BitPrefixFlag7 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag7 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        7
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag8;
impl BitPrefixFlag8 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag8 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        8
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag9;
impl BitPrefixFlag9 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag9 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        9
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

#[derive(Clone)]
pub struct BitPrefixFlag10;
impl BitPrefixFlag10 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag10 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        10
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct BitPrefixFlag11;
impl BitPrefixFlag11 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag11 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        11
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct BitPrefixFlag12;
impl BitPrefixFlag12 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag12 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        12
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct BitPrefixFlag13;
impl BitPrefixFlag13 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag13 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        13
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct BitPrefixFlag14;
impl BitPrefixFlag14 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for BitPrefixFlag14 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        14
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let constraint_idx = self.constraint_idx();

        let current_flag = current_step.get_evaluation_element(0, constraint_idx);
        let next_flag = current_step.get_evaluation_element(0, constraint_idx + 1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let bit = current_flag - two * next_flag;

        let res = bit * (bit - one);

        transition_evaluations[constraint_idx] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct ZeroFlagConstraint;
impl ZeroFlagConstraint {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for ZeroFlagConstraint {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        15
    }

    fn evaluate(
        &self,
        frame: &stark_platinum_prover::frame::Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let zero_flag = current_step.get_evaluation_element(0, 15);

        transition_evaluations[self.constraint_idx()] = *zero_flag;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct FlagOp1BaseOp0BitConstraint;
impl FlagOp1BaseOp0BitConstraint {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for FlagOp1BaseOp0BitConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        54
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let f_op1_imm = current_step.get_evaluation_element(0, 2);
        let f_op1_fp = current_step.get_evaluation_element(0, 3);
        let f_op1_ap = current_step.get_evaluation_element(0, 4);

        let f_op1_base_op0_bit = one - f_op1_imm - f_op1_fp - f_op1_ap;

        let res = f_op1_base_op0_bit * (f_op1_base_op0_bit - one);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct FlagResOp1BitConstraint;
impl FlagResOp1BitConstraint {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for FlagResOp1BitConstraint {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        55
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let f_res_add = current_step.get_evaluation_element(0, 5);
        let f_res_mul = current_step.get_evaluation_element(0, 6);
        let f_pc_jnz = current_step.get_evaluation_element(0, 9);

        let f_res_op1_bit = one - f_res_add - f_res_mul - f_pc_jnz;

        let res = f_res_op1_bit * (f_res_op1_bit - one);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct FlagPcUpdateRegularBit;
impl FlagPcUpdateRegularBit {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for FlagPcUpdateRegularBit {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        56
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let f_jump_abs = current_step.get_evaluation_element(0, 7);
        let f_jump_rel = current_step.get_evaluation_element(0, 8);
        let f_pc_jnz = current_step.get_evaluation_element(0, 9);

        let flag_pc_update_regular_bit = one - f_jump_abs - f_jump_rel - f_pc_jnz;

        let res = flag_pc_update_regular_bit * (flag_pc_update_regular_bit - one);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct FlagFpUpdateRegularBit;
impl FlagFpUpdateRegularBit {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for FlagFpUpdateRegularBit {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        57
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let f_opcode_call = current_step.get_evaluation_element(0, 12);
        let f_opcode_ret = current_step.get_evaluation_element(0, 13);

        let flag_fp_update_regular_bit = one - f_opcode_call - f_opcode_ret;

        let res = flag_fp_update_regular_bit * (flag_fp_update_regular_bit - one);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct InstructionUnpacking;
impl InstructionUnpacking {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for InstructionUnpacking {
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        16
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);

        let b15 = two.pow(15u32);
        let b16 = two.pow(16u32);
        let b32 = two.pow(32u32);
        let b48 = two.pow(48u32);

        // Named like this to match the Cairo whitepaper's notation.
        let f0_squiggle = current_step.get_evaluation_element(0, 0);

        let instruction = current_step.get_evaluation_element(0, 23);
        let off_dst = current_step.get_evaluation_element(0, 27);
        let off_op0 = current_step.get_evaluation_element(0, 28);
        let off_op1 = current_step.get_evaluation_element(0, 29);

        let res = off_dst + b16 * off_op0 + b32 * off_op1 + b48 * f0_squiggle - instruction;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesCallOff0;
impl CpuOpcodesCallOff0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesCallOff0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        58
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let f_opcode_call = current_step.get_evaluation_element(0, 12);
        let off_dst = current_step.get_evaluation_element(0, 27);

        let res = f_opcode_call * (off_dst - b15);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesCallOff1;
impl CpuOpcodesCallOff1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesCallOff1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        59
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let f_opcode_call = current_step.get_evaluation_element(0, 12);
        let off_op0 = current_step.get_evaluation_element(0, 28);

        let res = f_opcode_call * (off_op0 - b15 - one);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesCallFlags;
impl CpuOpcodesCallFlags {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesCallFlags {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        60
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let f_opcode_call = current_step.get_evaluation_element(0, 12);
        let flag0 = current_step.get_evaluation_element(0, 0);
        let flag1 = current_step.get_evaluation_element(0, 1);

        let res = f_opcode_call * (two * f_opcode_call + one + one - flag0 - flag1 - two - two);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesRetOff0;
impl CpuOpcodesRetOff0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesRetOff0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        61
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let f_opcode_ret = current_step.get_evaluation_element(0, 13);
        let off_dst = current_step.get_evaluation_element(0, 27);

        let res = f_opcode_ret * (off_dst + two - b15);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesRetOff2;
impl CpuOpcodesRetOff2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesRetOff2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        62
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let f_opcode_ret = current_step.get_evaluation_element(0, 13);
        let off_op1 = current_step.get_evaluation_element(0, 29);

        let res = f_opcode_ret * (off_op1 + one - b15);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesRetFlags;
impl CpuOpcodesRetFlags {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesRetFlags {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        63
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let f_opcode_ret = current_step.get_evaluation_element(0, 13);
        let flag0 = current_step.get_evaluation_element(0, 0);
        let flag3 = current_step.get_evaluation_element(0, 3);
        let flag7 = current_step.get_evaluation_element(0, 7);

        let f_res_add = current_step.get_evaluation_element(0, 5);
        let f_res_mul = current_step.get_evaluation_element(0, 6);
        let f_pc_jnz = current_step.get_evaluation_element(0, 9);

        let f_res_op1_bit = one - f_res_add - f_res_mul - f_pc_jnz;

        let res = f_opcode_ret * (flag7 + flag0 + flag3 + f_res_op1_bit - two - two);

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOperandsMemDstAddr;
impl CpuOperandsMemDstAddr {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOperandsMemDstAddr {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        17
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);
        let one = Felt252::one();
        let b15 = two.pow(15u32);
        let dst_fp = current_step.get_evaluation_element(0, 0)
            - two * current_step.get_evaluation_element(0, 1);
        let ap = current_step.get_evaluation_element(0, 17);
        let fp = current_step.get_evaluation_element(0, 18);
        let off_dst = current_step.get_evaluation_element(0, 27);
        let dst_addr = current_step.get_evaluation_element(0, 20);

        let res = dst_fp * fp + (one - dst_fp) * ap + (off_dst - b15) - dst_addr;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOperandsMem0Addr;
impl CpuOperandsMem0Addr {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOperandsMem0Addr {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        18
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);
        let one = Felt252::one();
        let b15 = two.pow(15u32);

        let op0_fp = current_step.get_evaluation_element(0, 1)
            - two * current_step.get_evaluation_element(0, 2);

        let ap = current_step.get_evaluation_element(0, 17);
        let fp = current_step.get_evaluation_element(0, 18);

        let off_op0 = current_step.get_evaluation_element(0, 28);
        let op0_addr = current_step.get_evaluation_element(0, 21);

        let res = op0_fp * fp + (one - op0_fp) * ap + (off_op0 - b15) - op0_addr;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOperandsMem1Addr;
impl CpuOperandsMem1Addr {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOperandsMem1Addr {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        19
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();
        let two = Felt252::from(2);
        let b15 = two.pow(15u32);

        let op1_val = current_step.get_evaluation_element(0, 2)
            - two * current_step.get_evaluation_element(0, 3);
        let op1_fp = current_step.get_evaluation_element(0, 3)
            - two * current_step.get_evaluation_element(0, 4);
        let op1_ap = current_step.get_evaluation_element(0, 4)
            - two * current_step.get_evaluation_element(0, 5);

        let op0 = current_step.get_evaluation_element(0, 25);
        let off_op1 = current_step.get_evaluation_element(0, 29);
        let op1_addr = current_step.get_evaluation_element(0, 22);

        let ap = current_step.get_evaluation_element(0, 17);
        let fp = current_step.get_evaluation_element(0, 18);
        let pc = current_step.get_evaluation_element(0, 19);

        let res = op1_val * pc
            + op1_ap * ap
            + op1_fp * fp
            + (one - op1_val - op1_ap - op1_fp) * op0
            + (off_op1 - b15)
            - op1_addr;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

// cpu/update_registers/update_ap/ap_update
pub struct CpuUpdateRegistersApUpdate;
impl CpuUpdateRegistersApUpdate {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuUpdateRegistersApUpdate {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        20
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let two = Felt252::from(2);

        let ap = current_step.get_evaluation_element(0, 17);
        let next_ap = next_step.get_evaluation_element(0, 17);
        let res = current_step.get_evaluation_element(0, 16);

        let ap_one = current_step.get_evaluation_element(0, 11)
            - two * current_step.get_evaluation_element(0, 12);
        let opc_call = current_step.get_evaluation_element(0, 12)
            - two * current_step.get_evaluation_element(0, 13);
        let ap_add = current_step.get_evaluation_element(0, 10)
            - two * current_step.get_evaluation_element(0, 11);

        let res = ap + ap_add * res + ap_one + opc_call * two - next_ap;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

pub struct CpuUpdateRegistersFpUpdate;
impl CpuUpdateRegistersFpUpdate {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuUpdateRegistersFpUpdate {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        21
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let ap = current_step.get_evaluation_element(0, 17);
        let fp = current_step.get_evaluation_element(0, 18);
        let next_fp = next_step.get_evaluation_element(0, 18);
        let dst = current_step.get_evaluation_element(0, 24);

        let ap_one = current_step.get_evaluation_element(0, 11)
            - two * current_step.get_evaluation_element(0, 12);
        let opc_call = current_step.get_evaluation_element(0, 12)
            - two * current_step.get_evaluation_element(0, 13);
        let opc_ret = current_step.get_evaluation_element(0, 13)
            - two * current_step.get_evaluation_element(0, 14);

        let res = opc_ret * dst + opc_call * (ap + two) + (one - opc_ret - opc_call) * fp - next_fp;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

// cpu/update_registers/update_pc/pc_cond_negative:
pub struct CpuUpdateRegistersPcCondNegative;
impl CpuUpdateRegistersPcCondNegative {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuUpdateRegistersPcCondNegative {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        23
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let one = Felt252::one();
        let two = Felt252::from(2);

        let t0 = current_step.get_evaluation_element(0, 30);
        let pc = current_step.get_evaluation_element(0, 19);
        let next_pc = next_step.get_evaluation_element(0, 19);
        let op1 = current_step.get_evaluation_element(0, 26);

        let pc_jnz = current_step.get_evaluation_element(0, 9)
            - two * current_step.get_evaluation_element(0, 10);
        let pc_abs = current_step.get_evaluation_element(0, 7)
            - two * current_step.get_evaluation_element(0, 8);
        let pc_rel = current_step.get_evaluation_element(0, 8)
            - two * current_step.get_evaluation_element(0, 9);
        let res = current_step.get_evaluation_element(0, 16);

        let res = t0 * (next_pc - (pc + op1)) + (one - pc_jnz) * next_pc
            - ((one - pc_abs - pc_rel - pc_jnz) * (pc + frame_inst_size(current_step))
                + pc_abs * res
                + pc_rel * (pc + res));
        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

pub struct CpuUpdateRegistersPcCondPositive;
impl CpuUpdateRegistersPcCondPositive {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuUpdateRegistersPcCondPositive {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        22
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let two = Felt252::from(2);

        let t1 = current_step.get_evaluation_element(0, 31);
        let pc_jnz = current_step.get_evaluation_element(0, 9)
            - two * current_step.get_evaluation_element(0, 10);
        let pc = current_step.get_evaluation_element(0, 19);
        let next_pc = next_step.get_evaluation_element(0, 19);

        let res = (t1 - pc_jnz) * (next_pc - (pc + frame_inst_size(current_step)));

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

//cpu/update_registers/update_pc/tmp0
pub struct CpuUpdateRegistersUpdatePcTmp0;
impl CpuUpdateRegistersUpdatePcTmp0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuUpdateRegistersUpdatePcTmp0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        24
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);
        let dst = current_step.get_evaluation_element(0, 24);
        let t0 = current_step.get_evaluation_element(0, 30);
        let pc_jnz = current_step.get_evaluation_element(0, 9)
            - two * current_step.get_evaluation_element(0, 10);

        let res = pc_jnz * dst - t0;

        transition_evaluations[self.constraint_idx()] = res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuUpdateRegistersUpdatePcTmp1;
impl CpuUpdateRegistersUpdatePcTmp1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuUpdateRegistersUpdatePcTmp1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        24
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);

        let t1 = current_step.get_evaluation_element(0, 31);

        let dst = current_step.get_evaluation_element(0, 24);
        let t0 = current_step.get_evaluation_element(0, 30);
        let pc_jnz = current_step.get_evaluation_element(0, 9)
            - two * current_step.get_evaluation_element(0, 10);
        let res = current_step.get_evaluation_element(0, 16);

        let transition_res = t0 * res - t1;

        transition_evaluations[self.constraint_idx()] = transition_res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOperandsOpsMul;
impl CpuOperandsOpsMul {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOperandsOpsMul {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        26
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let mul = current_step.get_evaluation_element(0, 32);
        let op0 = current_step.get_evaluation_element(0, 25);
        let op1 = current_step.get_evaluation_element(0, 26);

        transition_evaluations[self.constraint_idx()] = mul - op0 * op1;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

// cpu/operands/res
pub struct CpuOperandsRes;
impl CpuOperandsRes {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOperandsRes {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        27
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let one = Felt252::one();
        let two = Felt252::from(2);

        let mul = current_step.get_evaluation_element(0, 32);
        let op0 = current_step.get_evaluation_element(0, 25);
        let op1 = current_step.get_evaluation_element(0, 26);
        let res = current_step.get_evaluation_element(0, 16);

        let res_add = current_step.get_evaluation_element(0, 5)
            - two * current_step.get_evaluation_element(0, 6);
        let res_mul = current_step.get_evaluation_element(0, 6)
            - two * current_step.get_evaluation_element(0, 7);
        let pc_jnz = current_step.get_evaluation_element(0, 9)
            - two * current_step.get_evaluation_element(0, 10);

        let transition_res =
            res_add * (op0 + op1) + res_mul * mul + (one - res_add - res_mul - pc_jnz) * op1
                - (one - pc_jnz) * res;

        transition_evaluations[self.constraint_idx()] = transition_res;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

// cpu/opcodes/call/push_fp
pub struct CpuOpcodesCallPushFp;
impl CpuOpcodesCallPushFp {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesCallPushFp {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        28
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);

        let opc_call = current_step.get_evaluation_element(0, 12)
            - two * current_step.get_evaluation_element(0, 13);

        let dst = current_step.get_evaluation_element(0, 24);
        let fp = current_step.get_evaluation_element(0, 18);

        transition_evaluations[self.constraint_idx()] = opc_call * (dst - fp);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct CpuOpcodesCallPushPc;
impl CpuOpcodesCallPushPc {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesCallPushPc {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        29
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);

        let opc_call = current_step.get_evaluation_element(0, 12)
            - two * current_step.get_evaluation_element(0, 13);

        let op0 = current_step.get_evaluation_element(0, 25);
        let pc = current_step.get_evaluation_element(0, 19);

        transition_evaluations[self.constraint_idx()] =
            opc_call * (op0 - (pc + frame_inst_size(current_step)));
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

// cpu/opcodes/assert_eq/assert_eq
pub struct CpuOpcodesAssertEq;
impl CpuOpcodesAssertEq {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for CpuOpcodesAssertEq {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        30
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let two = Felt252::from(2);

        let opc_aeq = current_step.get_evaluation_element(0, 14)
            - two * current_step.get_evaluation_element(0, 15);
        let dst = current_step.get_evaluation_element(0, 24);
        let res = current_step.get_evaluation_element(0, 16);

        transition_evaluations[self.constraint_idx()] = opc_aeq * (dst - res)
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

// memory/diff_is_bit
pub struct MemoryDiffIsBit0;
impl MemoryDiffIsBit0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryDiffIsBit0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        31
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_0 = current_step.get_evaluation_element(0, 40);
        let mem_addr_sorted_1 = current_step.get_evaluation_element(0, 41);

        transition_evaluations[self.constraint_idx()] =
            (mem_addr_sorted_0 - mem_addr_sorted_1) * (mem_addr_sorted_1 - mem_addr_sorted_0 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryDiffIsBit1;
impl MemoryDiffIsBit1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryDiffIsBit1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        32
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_1 = current_step.get_evaluation_element(0, 41);
        let mem_addr_sorted_2 = current_step.get_evaluation_element(0, 42);

        transition_evaluations[self.constraint_idx()] =
            (mem_addr_sorted_1 - mem_addr_sorted_2) * (mem_addr_sorted_2 - mem_addr_sorted_1 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}
pub struct MemoryDiffIsBit2;
impl MemoryDiffIsBit2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryDiffIsBit2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        33
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_2 = current_step.get_evaluation_element(0, 42);
        let mem_addr_sorted_3 = current_step.get_evaluation_element(0, 43);

        transition_evaluations[self.constraint_idx()] =
            (mem_addr_sorted_2 - mem_addr_sorted_3) * (mem_addr_sorted_3 - mem_addr_sorted_2 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}
pub struct MemoryDiffIsBit3;
impl MemoryDiffIsBit3 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryDiffIsBit3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        34
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_3 = current_step.get_evaluation_element(0, 43);
        let mem_addr_sorted_4 = current_step.get_evaluation_element(0, 44);

        transition_evaluations[self.constraint_idx()] =
            (mem_addr_sorted_3 - mem_addr_sorted_4) * (mem_addr_sorted_4 - mem_addr_sorted_3 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}
pub struct MemoryDiffIsBit4;
impl MemoryDiffIsBit4 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryDiffIsBit4 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        35
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let one = Felt252::one();

        let next_mem_addr_sorted_0 = next_step.get_evaluation_element(0, 40);
        let mem_addr_sorted_4 = current_step.get_evaluation_element(0, 44);

        transition_evaluations[self.constraint_idx()] = (mem_addr_sorted_4
            - next_mem_addr_sorted_0)
            * (next_mem_addr_sorted_0 - mem_addr_sorted_4 - one);
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

// memory/is_func (single-valued)
pub struct MemoryIsFunc0;
impl MemoryIsFunc0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryIsFunc0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        36
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_0 = current_step.get_evaluation_element(0, 40);
        let mem_addr_sorted_1 = current_step.get_evaluation_element(0, 41);

        let mem_val_sorted_0 = current_step.get_evaluation_element(0, 45);
        let mem_val_sorted_1 = current_step.get_evaluation_element(0, 46);

        transition_evaluations[self.constraint_idx()] =
            (mem_val_sorted_0 - mem_val_sorted_1) * (mem_addr_sorted_1 - mem_addr_sorted_0 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryIsFunc1;
impl MemoryIsFunc1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryIsFunc1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        37
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_1 = current_step.get_evaluation_element(0, 41);
        let mem_addr_sorted_2 = current_step.get_evaluation_element(0, 42);

        let mem_val_sorted_1 = current_step.get_evaluation_element(0, 46);
        let mem_val_sorted_2 = current_step.get_evaluation_element(0, 47);

        transition_evaluations[self.constraint_idx()] =
            (mem_val_sorted_1 - mem_val_sorted_2) * (mem_addr_sorted_2 - mem_addr_sorted_1 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryIsFunc2;
impl MemoryIsFunc2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryIsFunc2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        38
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_2 = current_step.get_evaluation_element(0, 42);
        let mem_addr_sorted_3 = current_step.get_evaluation_element(0, 43);

        let mem_val_sorted_2 = current_step.get_evaluation_element(0, 47);
        let mem_val_sorted_3 = current_step.get_evaluation_element(0, 48);

        transition_evaluations[self.constraint_idx()] =
            (mem_val_sorted_2 - mem_val_sorted_3) * (mem_addr_sorted_3 - mem_addr_sorted_2 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryIsFunc3;
impl MemoryIsFunc3 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryIsFunc3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        39
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let one = Felt252::one();

        let mem_addr_sorted_3 = current_step.get_evaluation_element(0, 43);
        let mem_addr_sorted_4 = current_step.get_evaluation_element(0, 44);

        let mem_val_sorted_3 = current_step.get_evaluation_element(0, 48);
        let mem_val_sorted_4 = current_step.get_evaluation_element(0, 49);

        transition_evaluations[self.constraint_idx()] =
            (mem_val_sorted_3 - mem_val_sorted_4) * (mem_addr_sorted_4 - mem_addr_sorted_3 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}
pub struct MemoryIsFunc4;
impl MemoryIsFunc4 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryIsFunc4 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        40
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        _rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let one = Felt252::one();

        let next_mem_addr_sorted_0 = current_step.get_evaluation_element(0, 40);
        let mem_addr_sorted_4 = current_step.get_evaluation_element(0, 41);

        let next_mem_val_sorted_0 = current_step.get_evaluation_element(0, 45);
        let mem_val_sorted_4 = current_step.get_evaluation_element(0, 46);

        transition_evaluations[self.constraint_idx()] = (mem_val_sorted_4 - next_mem_val_sorted_0)
            * (next_mem_addr_sorted_0 - mem_addr_sorted_4 - one);
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

// memory/multi_column_perm/perm/step0
pub struct MemoryMultiColumnPermStep0_0;
impl MemoryMultiColumnPermStep0_0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryMultiColumnPermStep0_0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        41
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let alpha = rap_challenges[0];
        let z = rap_challenges[1];

        let p0 = current_step.get_evaluation_element(0, 50);
        let a1 = current_step.get_evaluation_element(0, 20);
        let ap1 = current_step.get_evaluation_element(0, 41);
        let vp1 = current_step.get_evaluation_element(0, 46);
        let p1 = current_step.get_evaluation_element(0, 51);
        let v1 = current_step.get_evaluation_element(0, 24);

        transition_evaluations[self.constraint_idx()] =
            (z - (ap1 + alpha * vp1)) * p1 - (z - (a1 + alpha * v1)) * p0;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryMultiColumnPermStep0_1;
impl MemoryMultiColumnPermStep0_1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryMultiColumnPermStep0_1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        42
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let alpha = rap_challenges[0];
        let z = rap_challenges[1];

        let p1 = current_step.get_evaluation_element(0, 51);
        let a2 = current_step.get_evaluation_element(0, 21);
        let ap2 = current_step.get_evaluation_element(0, 42);
        let vp2 = current_step.get_evaluation_element(0, 47);
        let p2 = current_step.get_evaluation_element(0, 52);
        let v2 = current_step.get_evaluation_element(0, 25);

        transition_evaluations[self.constraint_idx()] =
            (z - (ap2 + alpha * vp2)) * p2 - (z - (a2 + alpha * v2)) * p1;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryMultiColumnPermStep0_2;
impl MemoryMultiColumnPermStep0_2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryMultiColumnPermStep0_2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        43
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let alpha = rap_challenges[0];
        let z = rap_challenges[1];

        let p2 = current_step.get_evaluation_element(0, 52);
        let a3 = current_step.get_evaluation_element(0, 22);
        let ap3 = current_step.get_evaluation_element(0, 43);
        let vp3 = current_step.get_evaluation_element(0, 48);
        let p3 = current_step.get_evaluation_element(0, 53);
        let v3 = current_step.get_evaluation_element(0, 26);

        transition_evaluations[self.constraint_idx()] =
            (z - (ap3 + alpha * vp3)) * p3 - (z - (a3 + alpha * v3)) * p2;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryMultiColumnPermStep0_3;
impl MemoryMultiColumnPermStep0_3 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryMultiColumnPermStep0_3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        44
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let alpha = rap_challenges[0];
        let z = rap_challenges[1];

        let p3 = current_step.get_evaluation_element(0, 53);
        let p4 = current_step.get_evaluation_element(0, 54);
        let a4 = current_step.get_evaluation_element(0, 23);
        let v4 = current_step.get_evaluation_element(0, 27);
        let ap4 = current_step.get_evaluation_element(0, 44);
        let vp4 = current_step.get_evaluation_element(0, 49);

        transition_evaluations[self.constraint_idx()] =
            (z - (ap4 + alpha * vp4)) * p4 - (z - (a4 + alpha * v4)) * p3;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct MemoryMultiColumnPermStep0_4;
impl MemoryMultiColumnPermStep0_4 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for MemoryMultiColumnPermStep0_4 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        44
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let alpha = rap_challenges[0];
        let z = rap_challenges[1];

        let p4 = current_step.get_evaluation_element(0, 54);
        let next_v0 = current_step.get_evaluation_element(0, 23);
        let next_a0 = current_step.get_evaluation_element(0, 19);
        let next_p0 = current_step.get_evaluation_element(0, 50);
        let next_ap0 = next_step.get_evaluation_element(0, 40);
        let next_vp0 = current_step.get_evaluation_element(0, 45);

        transition_evaluations[self.constraint_idx()] =
            (z - (next_ap0 + alpha * next_vp0)) * next_p0 - (z - (next_a0 + alpha * next_v0)) * p4;
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

// rc16/diff_is_bit
pub struct Rc16DiffIsBit0;
impl Rc16DiffIsBit0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16DiffIsBit0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        46
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let one = Felt252::one();
        let z = rap_challenges[2];

        let rc_col_1 = current_step.get_evaluation_element(0, 36);
        let rc_col_2 = current_step.get_evaluation_element(0, 37);

        transition_evaluations[self.constraint_idx()] =
            (rc_col_1 - rc_col_2) * (rc_col_2 - rc_col_1 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct Rc16DiffIsBit1;
impl Rc16DiffIsBit1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16DiffIsBit1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        47
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let one = Felt252::one();
        let z = rap_challenges[2];

        let rc_col_2 = current_step.get_evaluation_element(0, 37);
        let rc_col_3 = current_step.get_evaluation_element(0, 38);

        transition_evaluations[self.constraint_idx()] =
            (rc_col_2 - rc_col_3) * (rc_col_3 - rc_col_2 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct Rc16DiffIsBit2;
impl Rc16DiffIsBit2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16DiffIsBit2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        48
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let one = Felt252::one();
        let z = rap_challenges[2];

        let rc_col_3 = current_step.get_evaluation_element(0, 38);
        let rc_col_4 = current_step.get_evaluation_element(0, 39);

        transition_evaluations[self.constraint_idx()] =
            (rc_col_3 - rc_col_4) * (rc_col_4 - rc_col_3 - one);
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct Rc16DiffIsBit3;
impl Rc16DiffIsBit3 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16DiffIsBit3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        49
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);
        let one = Felt252::one();
        let z = rap_challenges[2];

        let rc_col_4 = current_step.get_evaluation_element(0, 39);
        let next_rc_col_1 = next_step.get_evaluation_element(0, 36);

        transition_evaluations[self.constraint_idx()] =
            (rc_col_4 - next_rc_col_1) * (next_rc_col_1 - rc_col_4 - one);
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

// rc16/perm/step0
pub struct Rc16PermStep0_0;
impl Rc16PermStep0_0 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16PermStep0_0 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        50
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let z = rap_challenges[2];

        let ap1 = current_step.get_evaluation_element(0, 37);
        let p1 = current_step.get_evaluation_element(0, 56);
        let p0 = current_step.get_evaluation_element(0, 55);
        let a1 = current_step.get_evaluation_element(0, 28);

        transition_evaluations[self.constraint_idx()] = (z - ap1) * p1 - (z - a1) * p0;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct Rc16PermStep0_1;
impl Rc16PermStep0_1 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16PermStep0_1 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        51
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let z = rap_challenges[2];

        let ap2 = current_step.get_evaluation_element(0, 38);
        let p2 = current_step.get_evaluation_element(0, 57);
        let p1 = current_step.get_evaluation_element(0, 56);
        let a2 = current_step.get_evaluation_element(0, 29);

        transition_evaluations[self.constraint_idx()] = (z - ap2) * p2 - (z - a2) * p1;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct Rc16PermStep0_2;
impl Rc16PermStep0_2 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16PermStep0_2 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        52
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);

        let z = rap_challenges[2];

        let ap3 = current_step.get_evaluation_element(0, 39);
        let p3 = current_step.get_evaluation_element(0, 58);
        let p2 = current_step.get_evaluation_element(0, 57);
        let a3 = current_step.get_evaluation_element(0, 30);

        transition_evaluations[self.constraint_idx()] = (z - ap3) * p3 - (z - a3) * p2;
    }

    fn end_exemptions(&self) -> usize {
        0
    }
}

pub struct Rc16PermStep0_3;
impl Rc16PermStep0_3 {
    pub fn new() -> Self {
        Self
    }
}

impl TransitionConstraint<Stark252PrimeField> for Rc16PermStep0_3 {
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        53
    }

    fn evaluate(
        &self,
        frame: &Frame<Stark252PrimeField>,
        transition_evaluations: &mut [Felt252],
        _periodic_values: &[Felt252],
        rap_challenges: &[Felt252],
    ) {
        let current_step = frame.get_evaluation_step(0);
        let next_step = frame.get_evaluation_step(1);

        let z = rap_challenges[2];

        let p3 = current_step.get_evaluation_element(0, 58);

        let next_a0 = next_step.get_evaluation_element(0, 27);
        let next_ap0 = next_step.get_evaluation_element(0, 36);
        let next_p0 = next_step.get_evaluation_element(0, 55);

        transition_evaluations[self.constraint_idx()] =
            (z - next_ap0) * next_p0 - (z - next_a0) * p3;
    }

    fn end_exemptions(&self) -> usize {
        1
    }
}

fn frame_inst_size(step: &TableView<Stark252PrimeField>) -> Felt252 {
    let op1_val =
        step.get_evaluation_element(0, 2) - Felt252::from(2) * step.get_evaluation_element(0, 3);
    op1_val + Felt252::one()
}
