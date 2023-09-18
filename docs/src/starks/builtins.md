# Builtins

We can understand the built-in as a small machine, that we can use to efficiently prove a subprogram. For example, it may be able to prove a hash, like Poseidon or Keccak, verify a signature, or check that some variable is in a range, and the cost would be less than what we would have if using the Cairo VM instructions.

For each subprogram we want to prove, we will have a machine, which will have its own set of constraints in the prover. Let's take for example the Range Check built-in. This builtin enforces that a value $X$ is between 0 and $2^{128}$.

The logic behind the built-in is pretty straightforward. We split $X$ into 8 parts. So we will say that $X = X_{0} + X_{1} * 2^{16} + X_{2} * 2^{32} + X_{3} * 2^{48} + ... + X_{7} * 2^{112}$

Then we require that each is in the range $0 < X_{i} < 2^{16}$. The idea here is to reuse the Range Check constraint that checks if the offsets are between $0$ and $2^{16}$. If we can decompose the number in eight limbs of 16 bits, and we don't need any more limbs, it follows that the number will be less than $2^{128}$

The missing ingredient is how we make sure that each value $X$ that should be constrained by the built-in is actually constrained.

The process starts with the VM designating special memory positions for the built-in. You can think of this as a way of communicating the VM with the specific built-in machine by sharing memory. 

The VM won't save any instruction associated with how the built-in gets to the result and will assume the output is correct. You can think of this as an IO device in any computer, which works in a similar fashion. The VM delegates the work to an external device and takes the result from the memory.

Knowing which specific positions of the memory are used by the built-in, the prover can add more constraints that enforce the calculations of the built-in were done correctly. Let's see how it's done.

In the constraint system of the VM, we will treat every memory cell associated with the built-in as any other, treating it as a pair of addresses and values with the usual constraints. Additionally, we will add more that are specific to the builtin. 

Let's say we have multiple values $x_{i}$, such that each $x_{i}$ needs to be range checked by the built-in. Let each value be stored in a memory address $m_{i}$. Let the initial expected memory position for the range check built-in be $r_{0}$. Here $r_{0}$ is a value known and a public input.

We need to enforce then that $m_{0} = r_{0}$, and that the built in $m_{i+1} = m_{i} + 1$. These constraints have to be put on top of the constraints that are used by the memory, and that's the key to all of this. If these constraints weren't in place, there wouldn't be an enforced link between the Builtin and the VM, which would lead to security issues.

As one last detail, since the memory cells share the same constraints, and we add more for the ones in the builtin, we can treat the builtin cells as a subcolumn. In that case, we can assign one cell for the memory every N cell, giving a ratio that will be observable in the layout. 

This gives a better relationship between the number of cells used for the VM, and the builtin, giving an improvement in performance.
