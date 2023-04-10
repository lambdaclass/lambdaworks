# Cairo

To implement a prover for the Cairo programming language, we have to implement its `AIR`. So far, we have been dealing with simple toy examples where the computation is simple. Because we now have to implement a full-fledged virtual machine, a few new complexities arise; below we go through the main ones.

## High-level AIR description

The Cairo virtual machine uses a Von Neumann architecture with a Non-deterministic read-only memory. What this means is that the prover chooses all memory values when executing, and after that memory is immutable (i.e. you cannot write to it).

In practice, you can think of the `RAM` used by a program as a contiguous section of memory which can only grow. This way, managing memory is as simple as keeping a pointer to the next unused cell, increasing its value to allocate. This is similar to an arena or bump allocator, though here we never deallocate (memory can only be written to once).

There are only three registers in the Cairo VM:

- The program counter `pc`, which points to the next instruction to be executed.
- The allocation pointer `ap`, pointing to the next unused memory cell.
- The frame pointer `fp`, pointing to the base of the current stack frame. When a new function is called, `fp` is set to the current `ap`. When the function returns, `fp` goes back to its previous value. TODO: Talk about how dynamic memory allocation works here, as a bump allocator is sort of incompatible with a general heap allocator.

## Questions
- When running the cairo VM on a piece of code, aren't there supposed to be two output files? A memory one and a trace one or something?
- How do you allocate memory in cairo? Apart from automatic variables (i.e. function local ones), can you "heap allocate"? how does that work with segments? It seems like the User segment is the one used for heap allocation, though I'm not sure how that works.
