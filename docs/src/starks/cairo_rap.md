## Extended columns

The verifier sends challenges $\alpha, z \in \mathbb{F}$ (or the prover samples them from the transcript). Additional columns are added to incorporate the memory constraints. To define them the prover follows these steps:
1. Stack the rows of the submatrix of $T$ defined by the columns `pc, dst_addr, op0_addr, op1_addr` into a vector `a` of length $2^{n+2}$ (this means that the first entries of `a` are `pc[0], dst_addr[0], op0_addr[0], op1_addr[0], pc[1], dst_addr[1],...`).
1. Stack the the rows of the submatrix defined by the columns `inst, dst, op0, op1` into a vector `v` of length $2^{n+2}$.
1. Define $M_{\text{Mem}}\in\mathbb{F}^{2^{n+2}\times 2}$ to be the matrix with columns $a$, $v$.
1. Define $M_{\text{MemRepl}}\in\mathbb{F}^{2^{n+2}\times 2}$ to be the matrix that's equal to $M_{\text{Mem}}$ in the first $2^{n+2} - L_{\text{pub}}$ rows, and its last $L_{\text{pub}}$ entries are the addresses and values of the actual public memory (program code).
1. Sort $M_{\text{MemRepl}}$ by the first column in increasing order. The result is a matrix $M_{\text{MemReplSorted}}$ of size $2^{n+2}\times 2$. Denote its columns by $a'$ and $v'$.
1. Compute the vector $p$ of size $2^{n+2}$ with entries 
$$ p_i := \prod_{j=0}^i\frac{z - (a_i' + \alpha v_i')}{z - (a_i + \alpha v_i)}$$
1. Reshape the matrix $M_{\text{MemReplSorted}}$ into a $2^n\times 8$ in row-major. Reshape the vector $p$ into a $2^n \times 4$ matrix in row-major.
1. Concatenate these 12 rows. The result is a matrix $M_\text{MemRAP2}$ of size $2^n \times 12$

The verifier sends challenge $z' \in \mathbb{F}$. Further columns are added to incorporate the range check constraints following these steps:

1. Stack the rows of the submatrix of $T$ defined by the columns in the group `offsets` into a vector $b$ of length $3\cdot 2^n$.
1. Sort the values of $b$ in increasing order. Let $b'$ be the result.
1. Compute the vector $p'$ of size $3\cdot 2^n$ with entries
$$ p_i' := \prod_{j=0}^i\frac{z' - b_i'}{z' - b_i}$$
1. Reshape $b'$ and $p'$ into matrices of size $2^n \times 3$ each and concatenate them into a matrix $M_\text{RangeCheckRAP2}$ of size $2^n \times 6$.
1. Concatenate $M_\text{MemRAP2}$ and $M_\text{RangeCheckRAP2}$ into a matrix $M_\text{RAP2}$ of size $2^n \times 18$.


Using the notation described at the beginning, $m'=33$, $m''=18$ and $m=52$. They are respectively the columns of the first and second part of the rap, and the total number of columns.


Putting all together, the final layout of the trace is the following

```
 A.  flags      (16) : Decoded instruction flags
 B.  res        (1)  : Res value
 C.  pointers   (2)  : Temporary memory pointers (ap and fp)
 D.  mem_a      (4)  : Memory addresses (pc, dst_addr, op0_addr, op1_addr)
 E.  mem_v      (4)  : Memory values (inst, dst, op0, op1)
 F.  offsets    (3)  : (off_dst, off_op0, off_op1)
 G.  derived    (3)  : (t0, t1, mul)
 H.  mem_a'     (4)  : Sorted memory addresses
 I.  mem_v'     (4)  : Sorted memory values
 J.  mem_p      (4)  : Memory permutation argument columns
 K.  offsets_b' (3)  : Sorted offset columns
 L.  offsets_p' (3)  : Range check permutation argument columns

 A                B C  D    E    F   G   H    I    J    K   L
|xxxxxxxxxxxxxxxx|x|xx|xxxx|xxxx|xxx|xxx|xxxx|xxxx|xxxx|xxx|xxx|
```
