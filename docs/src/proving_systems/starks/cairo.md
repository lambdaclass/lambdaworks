# Cairo

To implement a prover for the Cairo programming language, we have to implement its `AIR`. So far, we have been dealing with simple toy examples where the computation is simple. Because we now have to implement a full-fledged virtual machine, a few new complexities arise; below we go through the main ones.

## High-level AIR description

The Cairo virtual machine uses a Von Neumann architecture with a Non-deterministic read-only memory. What this means is that the prover chooses all memory values when executing, and after that memory is immutable (i.e. you cannot write to it).

In practice, you can think of the `RAM` used by a program as a contiguous section of memory which can only grow. This way, managing memory is as simple as keeping a pointer to the next unused cell, increasing its value to allocate. This is similar to an arena or bump allocator, though here we never deallocate (memory can only be written to once).

There are only three registers in the Cairo VM:

- The program counter `pc`, which points to the next instruction to be executed.
- The allocation pointer `ap`, pointing to the next unused memory cell.
- The frame pointer `fp`, pointing to the base of the current stack frame. When a new function is called, `fp` is set to the current `ap`. When the function returns, `fp` goes back to its previous value. The VM creates new segments whenever dynamic allocation is needed, so for example the cairo analog to a `Vec` will have its own segment. Relocation at the end meshes everything together.


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


## Memory Constraints/Interaction step

In our regular STARK prover, the prover commits to the trace in a single step. In the Cairo prover, the commitment to the trace is done in two steps. The prover first commits to some columns, then adds a few more columns and commits to them afterwards.

We will explain what the memory constraints are to illustrate this interaction step.

As part of proving correct execution, the prover has to show that the memory used was indeed continuous and that it was only written to once. To do this, the prover takes the list of all the memory accesses that happened throughout execution. These are represented as pairs of elements `(address, value_at_address)`, of which there are four:

```
(pc, instruction_at_pc)
(dst_address, value_at_address)
(op0_address, op0)
(op1_address, op1)
```

Because memory accesses are scattered, proving that it's contiguous and written to only once is extremely complicated. If the list of memory accesses were *ordered by address*, it would be very simple: just prove that addresses are increasing and that every access to the same address gives the same value. To be more precise, if our list of accesses is this

$$
\{(a_1, v_1), (a_2, v_2), \dots, (a_n, v_n)\}
$$

and it's ordered, then all we need to do is show that:

- For every $i$, either $a_{i+1} = a_i$ or $a_{i + 1} = a_i + 1$ (memory is contiguous).
- For every pair $i \neq j$, if $a_i = a_j$ then $v_i = v_j$ (reads to the same address always return the same value).

Written as constraints, these are


- For every $i$:
    $$
    (a_{i+1} - a_i) (a_{i+1} - a_i - 1) = 0
    $$
- For every $i$:
    $$
    (v_{i+1} - v_i)(a_{i+1} - a_i - 1) = 0
    $$

With this idea in mind, to show the memory's consistency the prover does two things. First, they take all memory accesses and orders them. Second, they show that the above constraints are satisfied for the ordered list. If the ordered list is contiguous and write-once, then the unorderes once is too.

NOTE: If you're following along with the Cairo paper, the regular unordered list of memory accesses is what the paper calls $L_1$, while the *ordered* version of it is $L_2$.

This means two extra things are added to the protocol:

- The prover has to add the ordered list of memory accesses as a column to the trace, for which the constraints above apply.
- The verifier needs to check the constraints over those columns (i.e. those constraints need to be a part of the `AIR`).

We are not done yet, however. How does the verifier know that the list we proved the constraints for is indeed the result of ordering the memory accesses?. To show this, the prover has to perform what's called a *permutation argument*. A thorough explanation of what they are can be found [here](https://triton-vm.org/spec/permutation-argument.html); read it before continuing.

What the prover does, then, is commit to both the ordered and unordered list of memory accesses and then show one is a permutation of the other through a permutation argument. 

If you read the article above, you probably noticed that the argument is *interactive*. This means that the commitment to the trace has to be done in two steps. First a `main` trace is committed to, then an interactive step is done where the verifier provides some random values, and with them some extra `auxiliary` trace columns are added and committed to by the prover. Of course, in practice this step isn't actually interactive, we use Fiat-Shamir to simulate it.

Notice that, because the argument is interactive, the constraints associated to it *depend on the sampled Fiat-Shamir values*. This means that the associated `compute_transition` method will have to take fiat-shamir sampled values as arguments along with the frame. This is part of the reason why the trace, constraints, and other stuff are divided into `main` and `auxiliary`.

The explanation above was for the memory constraints. These are not the only constraints that require an interactive step and are therefore part of the `auxiliary` step. In general, constraints for builtins are also done during this step, as they also relate to memory.

## Public memory constraints

- What's public memory? Is it just the data segment? Public inputs? Something else?

## Permutation range-checks

Range check is a builtin, which means it's just data read from some place in memory. For it to make sense, the prover needs to prove that the space in memory dedicated to this builtin is indeed composed of numbers in the range $[0, 2^{16})$.

TODO: explain this, though because it's a builtin it's not the highest priority right now.

