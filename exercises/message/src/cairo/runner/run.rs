use super::vec_writer::VecWriter;
use crate::cairo::air::{MemorySegment, MemorySegmentMap, PublicInputs};
use crate::cairo::cairo_layout::CairoLayout;
use crate::cairo::cairo_mem::CairoMemory;
use crate::cairo::execution_trace::build_main_trace;
use crate::cairo::register_states::RegisterStates;
use crate::starks::trace::TraceTable;
use cairo_lang_starknet::casm_contract_class::CasmContractClass;
use cairo_vm::cairo_run::{self, EncodeTraceError};
use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor;
use cairo_vm::hint_processor::cairo_1_hint_processor::hint_processor::Cairo1HintProcessor;
use cairo_vm::serde::deserialize_program::BuiltinName;
use cairo_vm::types::{program::Program, relocatable::MaybeRelocatable};
use cairo_vm::vm::errors::{
    cairo_run_errors::CairoRunError, trace_errors::TraceError, vm_errors::VirtualMachineError,
};
use cairo_vm::vm::runners::cairo_runner::{CairoArg, CairoRunner, RunResources};
use cairo_vm::vm::vm_core::VirtualMachine;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Failed to interact with the file system")]
    IO(#[from] std::io::Error),
    #[error("The cairo program execution failed")]
    Runner(#[from] CairoRunError),
    #[error(transparent)]
    EncodeTrace(#[from] EncodeTraceError),
    #[error(transparent)]
    VirtualMachine(#[from] VirtualMachineError),
    #[error(transparent)]
    Trace(#[from] TraceError),
}

/// Indicates the version of the Cairo program.
/// This is used to determine how to parse and run the program.
pub enum CairoVersion {
    V0 = 0,
    V1 = 1,
}

/// Runs a cairo program in JSON format and returns trace, memory and program length.
/// Uses [cairo-rs](https://github.com/lambdaclass/cairo-rs/) project to run the program.
///
///  # Params
///
/// `entrypoint_function` - the name of the entrypoint function tu run. If `None` is provided, the default value is `main`.
/// `layout` - type of layout of Cairo.
/// `program_content` - content of the input file.
/// `trace_path` - path where to store the generated trace file.
/// `memory_path` - path where to store the generated memory file.
///
/// # Returns
///
/// Ok() in case of succes, with the following values:
/// - register_states
/// - cairo_mem
/// - data_len
/// - range_check: an Option<(usize, usize)> containing the start and end of range check.
/// `Error` indicating the type of error.
#[allow(clippy::type_complexity)]
pub fn run_program(
    entrypoint_function: Option<&str>,
    layout: CairoLayout,
    program_content: &[u8],
    cairo_version: &CairoVersion,
) -> Result<(RegisterStates, CairoMemory, usize, Option<Range<u64>>), Error> {
    // default value for entrypoint is "main"
    let entrypoint = entrypoint_function.unwrap_or("main");

    let args = [];

    let (vm, runner) = match cairo_version {
        CairoVersion::V0 => {
            let trace_enabled = true;
            let mut hint_executor = BuiltinHintProcessor::new_empty();
            let cairo_run_config = cairo_run::CairoRunConfig {
                entrypoint,
                trace_enabled,
                relocate_mem: true,
                layout: layout.as_str(),
                proof_mode: false,
                secure_run: None,
            };

            let (runner, vm) = match cairo_run::cairo_run(
                program_content,
                &cairo_run_config,
                &mut hint_executor,
            ) {
                Ok(runner) => runner,
                Err(error) => {
                    eprintln!("{error}");
                    return Err(Error::Runner(error));
                }
            };

            (vm, runner)
        }
        CairoVersion::V1 => {
            let casm_contract: CasmContractClass = serde_json::from_slice(program_content).unwrap();
            let program: Program = casm_contract.clone().try_into().unwrap();
            let mut runner = CairoRunner::new(
                &(casm_contract.clone().try_into().unwrap()),
                layout.as_str(),
                false,
            )
            .unwrap();
            let mut vm = VirtualMachine::new(true);

            runner
                .initialize_function_runner_cairo_1(&mut vm, &[BuiltinName::range_check])
                .unwrap();

            // Implicit Args
            let syscall_segment = MaybeRelocatable::from(vm.add_memory_segment());

            let builtins: Vec<&'static str> = runner
                .get_program_builtins()
                .iter()
                .map(|b| b.name())
                .collect();

            let builtin_segment: Vec<MaybeRelocatable> = vm
                .get_builtin_runners()
                .iter()
                .filter(|b| builtins.contains(&b.name()))
                .flat_map(|b| b.initial_stack())
                .collect();

            let initial_gas = MaybeRelocatable::from(usize::MAX);

            let mut implicit_args = builtin_segment;
            implicit_args.extend([initial_gas]);
            implicit_args.extend([syscall_segment]);

            // Other args

            // Load builtin costs
            let builtin_costs: Vec<MaybeRelocatable> =
                vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()];
            let builtin_costs_ptr = vm.add_memory_segment();
            vm.load_data(builtin_costs_ptr, &builtin_costs).unwrap();

            // Load extra data
            let core_program_end_ptr = (runner.program_base.unwrap() + program.data_len()).unwrap();
            let program_extra_data: Vec<MaybeRelocatable> =
                vec![0x208B7FFF7FFF7FFE.into(), builtin_costs_ptr.into()];
            vm.load_data(core_program_end_ptr, &program_extra_data)
                .unwrap();

            // Load calldata
            let calldata_start = vm.add_memory_segment();
            let calldata_end = vm.load_data(calldata_start, &args.to_vec()).unwrap();

            // Create entrypoint_args

            let mut entrypoint_args: Vec<CairoArg> = implicit_args
                .iter()
                .map(|m| CairoArg::from(m.clone()))
                .collect();
            entrypoint_args.extend([
                MaybeRelocatable::from(calldata_start).into(),
                MaybeRelocatable::from(calldata_end).into(),
            ]);
            let entrypoint_args: Vec<&CairoArg> = entrypoint_args.iter().collect();

            let mut hint_processor = Cairo1HintProcessor::new(&casm_contract.hints);

            // Run contract entrypoint
            // We assume entrypoint 0 for only one function
            let mut run_resources = RunResources::default();

            runner
                .run_from_entrypoint(
                    0,
                    &entrypoint_args,
                    &mut run_resources,
                    true,
                    Some(program.data_len() + program_extra_data.len()),
                    &mut vm,
                    &mut hint_processor,
                )
                .unwrap();

            let _ = runner.relocate(&mut vm, true);

            (vm, runner)
        }
    };

    let relocated_trace = vm.get_relocated_trace()?;

    let mut trace_vec = Vec::<u8>::new();
    let mut trace_writer = VecWriter::new(&mut trace_vec);
    trace_writer.write_encoded_trace(relocated_trace);

    let relocated_memory = &runner.relocated_memory;

    let mut memory_vec = Vec::<u8>::new();
    let mut memory_writer = VecWriter::new(&mut memory_vec);
    memory_writer.write_encoded_memory(relocated_memory);

    trace_writer.flush()?;
    memory_writer.flush()?;

    //TO DO: Better error handling
    let cairo_mem = CairoMemory::from_bytes_le(&memory_vec).unwrap();
    let register_states = RegisterStates::from_bytes_le(&trace_vec).unwrap();

    let data_len = runner.get_program().data_len();

    // get range start and end
    let range_check = vm
        .get_range_check_builtin()
        .map(|builtin| {
            let (idx, stop_offset) = builtin.get_memory_segment_addresses();
            let stop_offset = stop_offset.unwrap_or_default();
            let range_check_base =
                (0..idx).fold(1, |acc, i| acc + vm.get_segment_size(i).unwrap_or_default());
            let range_check_end = range_check_base + stop_offset;

            (range_check_base, range_check_end)
        })
        .ok();

    let range_check_builtin_range = range_check.map(|(start, end)| Range {
        start: start as u64,
        end: end as u64,
    });

    Ok((
        register_states,
        cairo_mem,
        data_len,
        range_check_builtin_range,
    ))
}

pub fn generate_prover_args(
    program_content: &[u8],
    cairo_version: &CairoVersion,
    output_range: &Option<Range<u64>>,
) -> Result<(TraceTable<Stark252PrimeField>, PublicInputs), Error> {
    let cairo_layout = match cairo_version {
        CairoVersion::V0 => CairoLayout::Small,
        CairoVersion::V1 => CairoLayout::Plain,
    };

    let (register_states, memory, program_size, range_check_builtin_range) =
        run_program(None, cairo_layout, program_content, cairo_version)?;

    let memory_segments = create_memory_segment_map(range_check_builtin_range, output_range);

    let mut pub_inputs =
        PublicInputs::from_regs_and_mem(&register_states, &memory, program_size, &memory_segments);

    let main_trace = build_main_trace(&register_states, &memory, &mut pub_inputs);

    Ok((main_trace, pub_inputs))
}

fn create_memory_segment_map(
    range_check_builtin_range: Option<Range<u64>>,
    output_range: &Option<Range<u64>>,
) -> MemorySegmentMap {
    let mut memory_segments = MemorySegmentMap::new();

    if let Some(range_check_builtin_range) = range_check_builtin_range {
        memory_segments.insert(MemorySegment::RangeCheck, range_check_builtin_range);
    }
    if let Some(output_range) = output_range {
        memory_segments.insert(MemorySegment::Output, output_range.clone());
    }

    memory_segments
}

pub fn cairo0_program_path(program_name: &str) -> String {
    const CARGO_DIR: &str = env!("CARGO_MANIFEST_DIR");
    const CAIRO0_BASE_REL_PATH: &str = "/cairo_programs/cairo0/";
    let program_base_path = CARGO_DIR.to_string() + CAIRO0_BASE_REL_PATH;
    program_base_path + program_name
}

pub fn cairo1_program_path(program_name: &str) -> String {
    const CARGO_DIR: &str = env!("CARGO_MANIFEST_DIR");
    const CAIRO1_BASE_REL_PATH: &str = "/cairo_programs/cairo1/";
    let program_base_path = CARGO_DIR.to_string() + CAIRO1_BASE_REL_PATH;
    program_base_path + program_name
}

#[cfg(test)]
mod tests {
    use crate::cairo::execution_trace::build_cairo_execution_trace;

    use super::*;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::MontgomeryConfigStark252PrimeField;
    use lambdaworks_math::field::{
        element::FieldElement, fields::montgomery_backed_prime_fields::U256PrimeField,
    };

    pub type Stark252PrimeField = U256PrimeField<MontgomeryConfigStark252PrimeField>;
    type FE = FieldElement<Stark252PrimeField>;

    #[test]
    fn test_parse_cairo_file() {
        let base_dir = env!("CARGO_MANIFEST_DIR");
        let json_filename = base_dir.to_owned() + "/src/cairo/runner/program.json";

        let program_content = std::fs::read(json_filename).unwrap();

        let (register_states, memory, program_size, _rg_in_out) = run_program(
            None,
            CairoLayout::AllCairo,
            &program_content,
            &CairoVersion::V0,
        )
        .unwrap();

        let pub_inputs = PublicInputs::from_regs_and_mem(
            &register_states,
            &memory,
            program_size,
            &MemorySegmentMap::new(),
        );

        let execution_trace = build_cairo_execution_trace(&register_states, &memory, &pub_inputs);

        // This trace is obtained from Giza when running the prover for the mentioned program.
        let expected_trace = TraceTable::new_from_cols(&[
            // col 0
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 1
            vec![FE::one(), FE::one(), FE::one()],
            // col 2
            vec![FE::one(), FE::one(), FE::zero()],
            // col 3
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 4
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 5
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 6
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 7
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 8
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 9
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 10
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 11
            vec![FE::one(), FE::zero(), FE::zero()],
            // col 12
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 13
            vec![FE::zero(), FE::zero(), FE::one()],
            // col 14
            vec![FE::one(), FE::one(), FE::zero()],
            // col 15
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 16
            vec![FE::from(3), FE::from(3), FE::from(9)],
            // col 17
            vec![FE::from(8), FE::from(9), FE::from(9)],
            // col 18
            vec![FE::from(8), FE::from(8), FE::from(8)],
            // col 19
            vec![FE::from(1), FE::from(3), FE::from(5)],
            // col 20
            vec![FE::from(8), FE::from(8), FE::from(6)],
            // col 21
            vec![FE::from(7), FE::from(7), FE::from(7)],
            // col 22
            vec![FE::from(2), FE::from(4), FE::from(7)],
            // col 23
            vec![
                FE::from(0x480680017fff8000),
                FE::from(0x400680017fff7fff),
                FE::from(0x208b7fff7fff7ffe),
            ],
            // col 24
            vec![FE::from(3), FE::from(3), FE::from(9)],
            // col 25
            vec![FE::from(9), FE::from(9), FE::from(9)],
            // col 26
            vec![FE::from(3), FE::from(3), FE::from(9)],
            // col 27
            vec![FE::from(0x8000), FE::from(0x7fff), FE::from(0x7ffe)],
            // col 28
            vec![FE::from(0x7fff), FE::from(0x7fff), FE::from(0x7fff)],
            // col 29
            vec![FE::from(0x8001), FE::from(0x8001), FE::from(0x7fff)],
            // col 30
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 31
            vec![FE::zero(), FE::zero(), FE::zero()],
            // col 32
            vec![FE::from(0x1b), FE::from(0x1b), FE::from(0x51)],
            // col 33 - Selector column
            vec![FE::one(), FE::one(), FE::zero()],
        ]);

        assert_eq!(execution_trace.cols(), expected_trace.cols());
    }
}
