# Cairo

To implement a prover for the Cairo programming language, we have to implement its `AIR`. So far, we have been dealing with simple toy examples where the computation is simple. Because we now have to implement a full-fledged virtual machine, a few new complexities arise; below we go through the main ones.

## High-level AIR description

The Cairo virtual machine uses a Von Neumann architecture with a Non-deterministic read-only memory. What this means is that the prover chooses all memory values when executing, and after that memory is immutable (i.e. you cannot write to it).

In practice, you can think of the `RAM` used by a program as a contiguous section of memory which can only grow. This way, managing memory is as simple as keeping a pointer to the next unused cell, increasing its value to allocate. This is similar to an arena or bump allocator, though here we never deallocate (memory can only be written to once).

There are only three registers in the Cairo VM:

- The program counter `pc`, which points to the next instruction to be executed.
- The allocation pointer `ap`, pointing to the next unused memory cell.
- The frame pointer `fp`, pointing to the base of the current stack frame. When a new function is called, `fp` is set to the current `ap`. When the function returns, `fp` goes back to its previous value. TODO: Talk about how dynamic memory allocation works here, as a bump allocator is sort of incompatible with a general heap allocator. The VM creates new segments whenever dynamic allocation is needed, so for example a `Vec` will have its own segment. Relocation at the end meshes everything together.


-------

- Program segment
- Execution segment
- User segment
- Builtin segment
- Instruction format: flags and offsets, high level overview of what they are used for.
- How state transitions are constructed from these instructions.
- There's a cairo assembly to abstract this as it's very complex.
- Builtins: range-check example. It's used to implement regular cpu arithmetic (i.e. module $2^64$).
- Public memory
- Virtual Columns
- Virtual subcolumns
- Take execution trace and memory
- 


# Cairo Constraints

## Instructions

## PC updates

This is a pretty complex constraint, as it is natively a cubic constraint. Because we want it to be quadratic, three new virtual columns with auxiliary variables `t_0`, `t_1` and `v` are introduced.

## Instruction Unpacking

Constraints that check that instructions are well formed (flags are all actual bits, offsets are in range, etc).

## Operand constraints

Constraints that check that the operands `op0_addr`, `op1_addr` and `dst_addr` are constructed accordingly from the instructions.

## `af` and `ap` registers constraints

Constraints to make sure that on each cycle, `af` and `ap` are updated accordingly.

## Opcodes and `res` constraints

To check opcodes and `res` are constructed correctly. TODO: explain what these are, especially `res`.

## Interaction step

In our regular STARK prover, the prover commits to the trace in a single step. In the Cairo prover, the commitment to the trace is done in two steps. The prover first commits to some columns before commiting to the rest. The reason for this is that some checks need to be performed, namely that some trace columns are permutations of others. This is done through what's called a *permutation argument*; they are very well explained [here](https://triton-vm.org/spec/permutation-argument.html).


I'm not sure if the permutation check needs to be done to create virtual columns from a set of (virtual) subcolumns or if it's something else.

## Memory constraints

When the prover passes the memory to the verifier, they have to prove that the memory is indeed contiguous and that it's read-only. Once they do that, memory *accesses* are added to the trace as virtual columns.

- What's public memory? Is it just the data segment? Public inputs? Something else?

## Permutation range-checks

Range check is a builtin, which means it's just data read from some place in memory. For it to make sense, the prover needs to prove that the space in memory dedicated to this builtin is indeed composed of numbers in the range $[0, 2^{16})$. 

