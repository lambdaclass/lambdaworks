# Cairo AIR execution trace - Giza

* The raw cairo execution trace is loaded from a trace file.
* A pre-trace (`State` in Giza) is computed for that trace and  memory files: 
    * from the register states obtained from the trace file and memory of the memory file, [register and instruction states are set](https://github.com/lambdaclass/giza/blob/8d997e12ea7102a45aa48523f3330a0955e380e4/runner/src/trace.rs#L238):
        * register states are set:
            * pc at mem_a[0]
            * ap at mem_p[0]
            * fp at mem_p[1]
        * instruction states are set:
            * res at res[0]
            * istruction at mem_v[0]
            * dst at mem_v[1]
            * op0 at mem_v[2]
            * op1 at mem_v[3]
            * dst_addr at mem_a[1]
            * op0_addr at mem_a[2]
            * op1_addr at mem_a[3]
            * off_dst at offsets[0]
            * off_op0 at offsets[1]
            * off_op1 at offsets[2]

* To build the ExecutionTrace, [3 more auxiliary columns are computed](https://github.com/lambdaclass/giza/blob/8d997e12ea7102a45aa48523f3330a0955e380e4/runner/src/trace.rs#L110):
    * t0: f_pc_jnz * dst
    * t1: t_0 * res
    * mul: op0 * op1
    The reason to this can be written in page 53 of the [Cairo whitepaper](https://eprint.iacr.org/2021/1063.pdf).

* [Compute memory holes and add dummy artificial memory accesses]():
    * get_memory_holes(): Gets memory accesses from state.mem_a, casts them to integers, sorts them and then iterates over them with a window of 2. In each window, substracts w[1] - w[0]. If the result is not 0 or 1, this means that there has been a memory address in the middle that has not been accessed. 
    **NOTE**: This may not be needed for the first iteration, since these are related to interaction with builtins

* Permutation range checks: As explained in section 9.9 of the [Cairo whitepaper](https://eprint.iacr.org/2021/1063.pdf): The offsets should be continuous so when holes are detected, they are added. See [here](https://github.com/lambdaclass/giza/blob/8d997e12ea7102a45aa48523f3330a0955e380e4/runner/src/trace.rs#L148). This works similarly as the get_memory_holes() function.

* [A selector column is added with 1s everywhere except for the last element of the trace](https://github.com/lambdaclass/giza/blob/8d997e12ea7102a45aa48523f3330a0955e380e4/runner/src/trace.rs#L183).

* [All columns are bundled together](https://github.com/lambdaclass/giza/blob/8d997e12ea7102a45aa48523f3330a0955e380e4/runner/src/trace.rs#L187).

Still need to understand better Virtual Columns and what role they play in the trace generation