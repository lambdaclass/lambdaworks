use std::ops::Range;

use crate::air::{MemorySegmentMap, PublicInputs, Segment, SegmentName};
use crate::cairo_layout::CairoLayout;
use crate::cairo_mem::CairoMemory;
use crate::execution_trace::build_main_trace;
use crate::register_states::RegisterStates;
use crate::Felt252;

use super::vec_writer::VecWriter;
use cairo_vm::cairo_run::{self, EncodeTraceError};

use cairo_vm::hint_processor::builtin_hint_processor::builtin_hint_processor_definition::BuiltinHintProcessor;

use cairo_vm::vm::errors::{
    cairo_run_errors::CairoRunError, trace_errors::TraceError, vm_errors::VirtualMachineError,
};

use cairo_vm::without_std::collections::HashMap;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::trace::TraceTable;

#[derive(Debug)]
pub enum Error {
    IO(std::io::Error),
    Runner(CairoRunError),
    EncodeTrace(EncodeTraceError),
    VirtualMachine(VirtualMachineError),
    Trace(TraceError),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::IO(err)
    }
}

impl From<CairoRunError> for Error {
    fn from(err: CairoRunError) -> Error {
        Error::Runner(err)
    }
}

impl From<EncodeTraceError> for Error {
    fn from(err: EncodeTraceError) -> Error {
        Error::EncodeTrace(err)
    }
}

impl From<VirtualMachineError> for Error {
    fn from(err: VirtualMachineError) -> Error {
        Error::VirtualMachine(err)
    }
}

impl From<TraceError> for Error {
    fn from(err: TraceError) -> Error {
        Error::Trace(err)
    }
}

fn from_vm_felt(f: cairo_vm::felt::Felt252) -> Felt252 {
    let val = f.to_biguint().to_str_radix(16);
    Felt252::from_hex(&val).unwrap()
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
) -> Result<(RegisterStates, CairoMemory, usize), Error> {
    // default value for entrypoint is "main"
    let entrypoint = entrypoint_function.unwrap_or("main");

    let trace_enabled = true;
    let mut hint_executor = BuiltinHintProcessor::new_empty();
    let cairo_run_config = cairo_run::CairoRunConfig {
        entrypoint,
        trace_enabled,
        relocate_mem: true,
        layout: layout.as_str(),
        proof_mode: true,
        secure_run: None,
        disable_trace_padding: false,
    };

    let (runner, vm) =
        match cairo_run::cairo_run(program_content, &cairo_run_config, &mut hint_executor) {
            Ok(runner) => runner,
            Err(error) => {
                eprintln!("{error}");
                panic!();
            }
        };

    let relocated_trace = vm.get_relocated_trace().unwrap();

    let mut trace_vec = Vec::<u8>::new();
    let mut trace_writer = VecWriter::new(&mut trace_vec);
    trace_writer.write_encoded_trace(relocated_trace);

    let relocated_memory = &runner.relocated_memory;

    let mut memory_vec = Vec::<u8>::new();
    let mut memory_writer = VecWriter::new(&mut memory_vec);
    memory_writer.write_encoded_memory(relocated_memory);

    trace_writer.flush().unwrap();
    memory_writer.flush().unwrap();

    //TO DO: Better error handling
    let cairo_mem = CairoMemory::from_bytes_le(&memory_vec).unwrap();
    let register_states = RegisterStates::from_bytes_le(&trace_vec).unwrap();

    let data_len = runner.get_program().data_len();

    let range_check_builtin_included = runner
        .get_program()
        .iter_builtins()
        .any(|builtin| builtin.name() == "range_check_builtin");

    // get range start and end
    let range_check = if range_check_builtin_included {
        vm.get_range_check_builtin()
            .map(|builtin| {
                let (idx, stop_offset) = builtin.get_memory_segment_addresses();
                let stop_offset = stop_offset.unwrap_or_default();
                let range_check_base =
                    (0..idx).fold(1, |acc, i| acc + vm.get_segment_size(i).unwrap_or_default());
                let range_check_end = range_check_base + stop_offset;

                (range_check_base, range_check_end)
            })
            .ok()
    } else {
        None
    };

    let range_check_builtin_range = range_check.map(|(start, end)| Range {
        start: start as u64,
        end: end as u64,
    });

    // let vm_public_inputs = runner.get_air_public_input(&vm).unwrap();
    // let memory_segments =
    //     create_memory_segment_map(range_check_builtin_range.clone(), output_range);

    // let public_memory = vm_public_inputs.public_memory.iter().fold(
    //     HashMap::<Felt252, Felt252>::new(),
    //     |mut acc, public_memory_entry| {
    //         if let Some(v) = &public_memory_entry.value {
    //             // ! public_memory_entry.page is not used
    //             acc.insert(
    //                 Felt252::from(public_memory_entry.address as u64),
    //                 Felt252::from(from_vm_felt(v.clone())),
    //             );
    //         } else {
    //             panic!();
    //         }
    //         acc
    //     },
    // );

    // let num_steps = register_states.steps();
    // let public_inputs = PublicInputs {
    //     pc_init: Felt252::from(register_states.rows[0].pc),
    //     ap_init: Felt252::from(register_states.rows[0].ap),
    //     fp_init: Felt252::from(register_states.rows[0].fp),
    //     pc_final: Felt252::from(register_states.rows[num_steps - 1].pc),
    //     ap_final: Felt252::from(register_states.rows[num_steps - 1].ap),
    //     range_check_min: Some(vm_public_inputs.rc_min as u16),
    //     range_check_max: Some(vm_public_inputs.rc_max as u16),
    //     memory_segments,
    //     public_memory,
    //     num_steps,
    //     codelen: data_len,
    // };

    Ok((
        register_states,
        cairo_mem,
        data_len,
        // public_inputs,
    ))
}

pub fn generate_prover_args(
    program_content: &[u8],
    layout: CairoLayout,
) -> Result<(TraceTable<Stark252PrimeField>, PublicInputs), Error> {
    let (register_states, memory, program_size) = run_program(None, layout, program_content)?;

    let mut pub_inputs = PublicInputs::from_regs_and_mem(&register_states, &memory, program_size);

    let main_trace = build_main_trace(&register_states, &memory, &mut pub_inputs);

    // println!(
    //     "min\nGot {}, Expected {}",
    //     public_inputs.range_check_min.unwrap(),
    //     pub_inputs.range_check_min.unwrap()
    // );

    // // testing
    // pub_inputs.pc_init = public_inputs.pc_init;
    // pub_inputs.ap_init = public_inputs.ap_init;
    // pub_inputs.fp_init = public_inputs.fp_init;
    // pub_inputs.pc_final = public_inputs.pc_final;
    // pub_inputs.ap_final = public_inputs.ap_final;
    // // pub_inputs.range_check_min = public_inputs.range_check_min; // breaks
    // pub_inputs.range_check_max = public_inputs.range_check_max;
    // pub_inputs.memory_segments = public_inputs.memory_segments;
    // // pub_inputs.public_memory = public_inputs.public_memory; // breaks
    // pub_inputs.num_steps = public_inputs.num_steps;
    // pub_inputs.codelen = public_inputs.codelen;

    Ok((main_trace, pub_inputs))
}

pub fn generate_prover_args_from_trace(
    trace_bin_path: &str,
    memory_bin_path: &str,
) -> Result<(TraceTable<Stark252PrimeField>, PublicInputs), Error> {
    // ## Generating the prover args
    let register_states =
        RegisterStates::from_file(trace_bin_path).expect("Cairo trace bin file not found");
    let memory =
        CairoMemory::from_file(memory_bin_path).expect("Cairo memory binary file not found");

    // data length
    let data_len = 0_usize;
    let mut pub_inputs = PublicInputs::from_regs_and_mem(&register_states, &memory, data_len);

    let main_trace = build_main_trace(&register_states, &memory, &mut pub_inputs);

    Ok((main_trace, pub_inputs))
}
