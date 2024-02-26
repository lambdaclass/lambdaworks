import json
import os
from typing import Any, List, Union

import aiofiles

from starkware.cairo.bootloaders.fact_topology import (
    FactTopologiesFile,
    FactTopology,
    get_fact_topology_from_additional_data,
)
from starkware.cairo.bootloaders.simple_bootloader.objects import CairoPieTask, RunProgramTask, Task
from starkware.cairo.common.hash_state import compute_hash_on_elements
from starkware.cairo.lang.builtins.all_builtins import ALL_BUILTINS
from starkware.cairo.lang.compiler.program import Program
from starkware.cairo.lang.vm.cairo_pie import CairoPie, ExecutionResources
from starkware.cairo.lang.vm.output_builtin_runner import OutputBuiltinRunner
from starkware.cairo.lang.vm.relocatable import MaybeRelocatable, RelocatableValue, relocate_value
from starkware.python.utils import WriteOnceDict, from_bytes

SIMPLE_BOOTLOADER_COMPILED_PATH = os.path.join(
    os.path.dirname(__file__), "simple_bootloader_compiled.json"
)

# Upper bounds on the numbers of builtin instances and steps that the simple_bootloader uses.
SIMPLE_BOOTLOADER_N_OUTPUT = 2
SIMPLE_BOOTLOADER_N_PEDERSEN = 20
SIMPLE_BOOTLOADER_N_RANGE_CHECKS = 20
SIMPLE_BOOTLOADER_N_STEPS_CONSTANT = 400
SIMPLE_BOOTLOADER_N_STEPS_RATIO = 8


async def get_simple_bootloader_program_json() -> str:
    async with aiofiles.open(SIMPLE_BOOTLOADER_COMPILED_PATH, "r") as file:
        return json.loads(await file.read())


async def get_simple_bootloader_program() -> Program:
    async with aiofiles.open(SIMPLE_BOOTLOADER_COMPILED_PATH, "r") as file:
        return Program.Schema().loads(await file.read())


async def get_simple_bootloader_program_hash() -> int:
    """
    Returns the hash of the simple bootloader program. Matches the Cairo verifier's expected simple
    bootloader hash.
    """
    simple_bootloader_program: Program = await get_simple_bootloader_program()
    return compute_hash_on_elements(data=simple_bootloader_program.data)


def load_program(task: Task, memory, program_header, builtins_offset):
    """
    Fills the memory with the following:
    1. program header.
    2. program code.
    Returns the program address and the size of the written program data.
    """

    builtins = task.get_program().builtins
    n_builtins = len(builtins)
    program_data = task.get_program().data

    # Fill in the program header.
    header_address = program_header.address_
    # The program header ends with a list of builtins used by the program.
    header_size = builtins_offset + n_builtins
    # data_length does not include the data_length header field in the calculation.
    program_header.data_length = (header_size - 1) + len(program_data)
    program_header.program_main = task.get_program().main
    program_header.n_builtins = n_builtins
    # Fill in the builtin list in memory.
    builtins_address = header_address + builtins_offset
    for index, builtin in enumerate(builtins):
        assert isinstance(builtin, str)
        memory[builtins_address + index] = from_bytes(builtin.encode("ascii"))

    # Fill in the program code in memory.
    program_address = header_address + header_size
    for index, opcode in enumerate(program_data):
        memory[program_address + index] = opcode

    return program_address, header_size + len(program_data)


def write_return_builtins(
    memory,
    return_builtins_addr,
    used_builtins,
    used_builtins_addr,
    pre_execution_builtins_addr,
    task,
):
    """
    Writes the updated builtin pointers after the program execution to the given return builtins
    address.
    used_builtins is the list of builtins used by the program and thus updated by it.
    """

    used_builtin_offset = 0
    for index, builtin in enumerate(ALL_BUILTINS):
        if builtin in used_builtins:
            memory[return_builtins_addr + index] = memory[used_builtins_addr + used_builtin_offset]
            used_builtin_offset += 1

            if isinstance(task, CairoPie):
                assert task.metadata.builtin_segments[builtin].size == (
                    memory[return_builtins_addr + index]
                    - memory[pre_execution_builtins_addr + index]
                ), "Builtin usage is inconsistent with the CairoPie."
        else:
            # The builtin is unused, hence its value is the same as before calling the program.
            memory[return_builtins_addr + index] = memory[pre_execution_builtins_addr + index]


def load_cairo_pie(
    task: CairoPie,
    memory,
    segments,
    program_address,
    execution_segment_address,
    builtin_runners,
    ret_fp,
    ret_pc,
):
    """
    Load memory entries of the inner program.
    This replaces executing hints in a non-trusted program.
    """
    segment_offsets = WriteOnceDict()

    segment_offsets[task.metadata.program_segment.index] = program_address
    segment_offsets[task.metadata.execution_segment.index] = execution_segment_address
    segment_offsets[task.metadata.ret_fp_segment.index] = ret_fp
    segment_offsets[task.metadata.ret_pc_segment.index] = ret_pc

    def extract_segment(value: MaybeRelocatable, value_name: str):
        """
        Returns the segment index for the given value.
        Verifies that value is a RelocatableValue with offset 0.
        """
        assert isinstance(value, RelocatableValue), f"{value_name} is not relocatable."
        assert value.offset == 0, f"{value_name} has a non-zero offset."
        return value.segment_index

    orig_execution_segment = RelocatableValue(
        segment_index=task.metadata.execution_segment.index, offset=0
    )

    # Set initial stack relocations.
    for idx, name in enumerate(task.program.builtins):
        segment_offsets[
            extract_segment(
                value=task.memory[orig_execution_segment + idx],
                value_name=f"{name} builtin start address",
            )
        ] = memory[execution_segment_address + idx]

    for segment_info in task.metadata.extra_segments:
        segment_offsets[segment_info.index] = segments.add(size=segment_info.size)

    def local_relocate_value(value):
        return relocate_value(value, segment_offsets, task.program.prime)

    # Relocate builtin additional data.
    # This should occur before the memory relocation, since the signature builtin assumes that a
    # signature is added before the corresponding public key and message are both written to memory.
    esdsa_additional_data = task.additional_data.get("ecdsa_builtin")
    if esdsa_additional_data is not None:
        ecdsa_builtin = builtin_runners.get("ecdsa_builtin")
        assert ecdsa_builtin is not None, "The task requires the ecdsa builtin but it is missing."
        ecdsa_builtin.extend_additional_data(esdsa_additional_data, local_relocate_value)

    for addr, val in task.memory.items():
        memory[local_relocate_value(addr)] = local_relocate_value(val)


def prepare_output_runner(
    task: Task, output_builtin: OutputBuiltinRunner, output_ptr: RelocatableValue
):
    """
    Prepares the output builtin if the type of task is Task, so that pages of the inner program
    will be recorded separately.
    If the type of task is CairoPie, nothing should be done, as the program does not contain
    hints that may affect the output builtin.
    The return value of this function should be later passed to get_task_fact_topology().
    """

    if isinstance(task, RunProgramTask):
        output_state = output_builtin.get_state()
        output_builtin.new_state(base=output_ptr)
        return output_state
    elif isinstance(task, CairoPieTask):
        return None
    else:
        raise NotImplementedError(f"Unexpected task type: {type(task).__name__}.")


def get_task_fact_topology(
    output_size: int,
    task: Union[RunProgramTask, CairoPie],
    output_builtin: OutputBuiltinRunner,
    output_runner_data: Any,
) -> FactTopology:
    """
    Returns the fact_topology that corresponds to 'task'. Restores output builtin state if 'task' is
    a RunProgramTask.
    """

    # Obtain the fact_toplogy of 'task'.
    if isinstance(task, RunProgramTask):
        assert output_runner_data is not None
        fact_topology = get_fact_topology_from_additional_data(
            output_size=output_size,
            output_builtin_additional_data=output_builtin.get_additional_data(),
        )
        # Restore the output builtin runner to its original state.
        output_builtin.set_state(output_runner_data)
    elif isinstance(task, CairoPieTask):
        assert output_runner_data is None
        fact_topology = get_fact_topology_from_additional_data(
            output_size=output_size,
            output_builtin_additional_data=task.cairo_pie.additional_data["output_builtin"],
        )
    else:
        raise NotImplementedError(f"Unexpected task type: {type(task).__name__}.")

    return fact_topology


def add_consecutive_output_pages(
    fact_topology: FactTopology,
    output_builtin: OutputBuiltinRunner,
    cur_page_id: int,
    output_start: MaybeRelocatable,
) -> int:
    offset = 0
    for i, page_size in enumerate(fact_topology.page_sizes):
        output_builtin.add_page(
            page_id=cur_page_id + i, page_start=output_start + offset, page_size=page_size
        )
        offset += page_size

    return len(fact_topology.page_sizes)


def configure_fact_topologies(
    fact_topologies: List[FactTopology],
    output_start: MaybeRelocatable,
    output_builtin: OutputBuiltinRunner,
):
    """
    Given the fact_topologies of the tasks that were run by bootloader, configure the
    corresponding pages in the output builtin. Assumes that the bootloader output 2 words per task.
    """
    # Each task may use a few memory pages. Start from page 1 (as page 0 is reserved for the
    # bootloader program and arguments).
    cur_page_id = 1
    for fact_topology in fact_topologies:
        # Skip bootloader output for each task.
        output_start += 2
        cur_page_id += add_consecutive_output_pages(
            fact_topology=fact_topology,
            output_builtin=output_builtin,
            cur_page_id=cur_page_id,
            output_start=output_start,
        )
        output_start += sum(fact_topology.page_sizes)


def write_to_fact_topologies_file(fact_topologies_path: str, fact_topologies: List[FactTopology]):
    with open(fact_topologies_path, "w") as fp:
        json.dump(
            FactTopologiesFile.Schema().dump(FactTopologiesFile(fact_topologies=fact_topologies)),
            fp,
            indent=4,
            sort_keys=True,
        )
        fp.write("\n")


def calc_simple_bootloader_execution_resources(program_length: int) -> ExecutionResources:
    """
    Returns an upper bound on the number of steps and builtin instances that the simple bootloader
    uses.
    """
    n_steps = SIMPLE_BOOTLOADER_N_STEPS_RATIO * program_length + SIMPLE_BOOTLOADER_N_STEPS_CONSTANT
    builtin_instance_counter = {
        "pedersen_builtin": SIMPLE_BOOTLOADER_N_PEDERSEN + program_length,
        "range_check_builtin": SIMPLE_BOOTLOADER_N_RANGE_CHECKS,
        "output_builtin": SIMPLE_BOOTLOADER_N_OUTPUT,
    }
    return ExecutionResources(
        n_steps=n_steps, builtin_instance_counter=builtin_instance_counter, n_memory_holes=0
    )
