# Trace

The execution of a Cairo program produces a memory vector $V$ and a matrix $M$ of size $L \times 3$ with the evolution of the three registers `pc`, `ap`, `fp`. All of them with entries in $\mathbb{F}$.

## Construction of execution trace $T$:
In this section we describe the construction of the execution trace $T$. This is the matrix mentioned [here](#definitions) in the description of the STARK protocol

1. Augment each row of $M$ with information about the pointed instruction as follows: For each entry $(\text{pc}_i, \text{ap}_i, \text{fp}_i)$ of $M$, unpack the $\text{pc}_i$-th value of $V$. The result is a new matrix $M \in \mathbb{F}^{L\times 33}$ with the following layout
```
 A.  flags     (16) : Decoded instruction flags
 B.  res       (1)  : Res value
 C.  pointers  (2)  : Temporary memory pointers (ap and fp)
 D.  mem_a     (4)  : Memory addresses (pc, dst_addr, op0_addr, op1_addr)
 E.  mem_v     (4)  : Memory values (inst, dst, op0, op1)
 F.  offsets   (3)  : (off_dst, off_op0, off_op1)
 G.  derived   (3)  : (t0, t1, mul)

 A                B C  D    E    F   G
|xxxxxxxxxxxxxxxx|x|xx|xxxx|xxxx|xxx|xxx|
```
1. Let $r_\text{min}$ and $r_\text{max}$ be respectively the minimum and maximum values of the entries of the submatrix $M_\text{offsets}$ defined by the columns of the group `offsets`. Let $v$ be the vector of all the values between $r_\text{min}$ and $r_\text{max}$ that are not in $M_\text{offsets}$. If the length of $v$ is not a multiple of three, extend it to the nearest multiple of three using one arbitrary value of $v$.

1. Let $R$ be the last row of $M$, and let $R'$ be the vector that's equal to $R$ except that it has zeroes in entries corresponding to the ordered set of columns `mem_a` and `mem_v`. The set is ordered incrementally by `mem_a`. Let $L_{\text{pub}}$ be the length of the public input (program code). Extend $M$ with additional $L':=\lceil L_{\text{pub}}/4 \rceil$ rows to obtain a matrix $M \in \mathbb{F}^{(L + L')\times 33}$ by appending copies of $R'$ at the bottom (the notation $\lceil x \rceil$ means the _ceiling function_, defined as the smallest integer that is not smaller than $x$).

1. Let $R''$ be the vector that's equal to $R$ except that it has zeroes in entries corresponding to the set of columns `mem_a` and `mem_v`, let $M_\text{addr}$ be the submatrix defined by the columns of the group `addresses`, let $L'' = (L''_0, L''_1, ..., L''_J)^T$ the submatrix that asserts $M_\text{addr,i,j} < L''_\text{0,j}$, $L''_\text{I,j} < M_\text{addr,i+1,j}$ where $M_\text{addr,i+1,j} - M_\text{addr,i,j} > 1$ and $0 \le j \le J$ and $I = |L''_j|$. Extend $M$ with additional $L''$ rows to obtain a matrix $M \in \mathbb{F}^{(L + L' + L'')\times 33}$ by appending copies of $R''$ at the bottom.

1. Pad $M$ with copies of its last row until it has a power of two number of rows. As a result we obtain a matrix $T\in\mathbb{F}^{2^n\times 33}$.
