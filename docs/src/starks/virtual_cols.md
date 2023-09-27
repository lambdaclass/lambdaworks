# Virtual columns and Subcolumns
## Virtual Columns

In previous chapters, we have seen how the registers states and the memory are augmented to generate a provable trace. 

While we have shown a way of doing that, there isn't only one possible provable trace. In fact, there are multiple configurations possible. 

For example, in the Cairo VM, we have 15 flags. These flags include  "DstReg", "Op0Reg", "OpCode" and others. For simplification, let's imagine we have 3 flags with letters from "A" to "C", where "A" is the first flag. 

Now, let's assume we have 4 steps in our trace. If we were to only use plain columns, the layout would look like this:

| FlagA| FlagB| FlagB|
|  --  |  --  | --   |
|  A0  |  B0  |  C0  |
|  A1  |  B1  |  C1  |
|  A2  |  B2  |  C2  |
|  A3  |  B3  |  C3  |

But, we could also organize them like this

| Flags|
|  --  |
|  A0  |
|  B0  |
|  C0  |
|  A1  |
|  B1  |
|  C1  |
|  A2  |
|  B2  |
|  C2  |
|  A3  |
|  B3  |
|  C3  |

The only problem is that now the constraints for each transition of the rows are not the same. We will have to define then a concept called "Virtual Column".

A Virtual Column is like a traditional column, which has its own set of constraints, but it exists interleaved with another one. In the previous example, each row is associated with a column, but in practice, we could have different ratios. We could have 3 rows corresponding to one Virtual Column, and the next one corresponding to another one. For the time being, let's focus on this simpler example.

Each row corresponding to Flag A will have the constraints associated with its own Virtual Column, and the same will apply to Flag B and Flag C.

Now, to do this, we will need to evaluate the multiple rows taking into account that they are part of the same step. For a real case, we will add a dummy flag D, whose purpose is to make the evaluation move in a number that is a power of 2. 

Let's see how it works. If we were evaluating the Frame where the constraints should give 0, the frame movement would look like this:

```diff
+ A0 | B0 | C0
+ A1 | B1 | C1
  A2 | B2 | C2
  A3 | B3 | C3
```
```diff
  A0 | B0 | C0
+ A1 | B1 | C1
+ A2 | B2 | C2
  A3 | B3 | C3
```
```diff
  A0 | B0 | C0
  A1 | B1 | C1
+ A2 | B2 | C2
+ A3 | B3 | C3
```

In the second case, the evaluation would look like this:

```diff 
+ A0 |
+ B0 |
+ C0 |
+ D0 |
+ A1 |
+ B1 |
+ C1 |
+ D1 |
  A2 |
  B2 |
  C2 |
  D2 |
  A3 |
  B3 |
  C3 |
  D3 |
```
```diff
  A0 |
  B0 |
  C0 |
  D0 |
+ A1 |
+ B1 |
+ C1 |
+ D1 |
+ A2 |
+ B2 |
+ C2 |
+ D2 |
  A3 |
  B3 |
  C3 |
  D3 |
```

```diff
  A0 |
  B0 |
  C0 |
  D0 |
  A1 |
  B1 |
  C1 |
  D1 |
+ A2 |
+ B2 |
+ C2 |
+ D2 |
+ A3 |
+ B3 |
+ C3 |
+ D3 |
```

When evaluating the composition polynomial, we will do it over the points on the LDE, where the constraints won't evaluate to 0, but we will use the same spacing. Assume we have three constraints for each flag, $C_{0}$, $C_{1}$, and $C_{2}$, and that they don't involve other trace cells. Let's call the index of the frame evaluation i, starting from 0.

In the first case, the constraint $C_{0}$, $C_{1}$ and $C_{2}$ would be applied over the same rows, giving an equation that looks like this:

$`C_{k}(w^i, w^{i+1})`$

In the second case, the equations would look like:

$`C_{0}(w^{i*4}, w^{i*4+4})`$

$`C_{1}(w^{i*4+1}, w^{i*4+5})`$

$`C_{2}(w^{i*4+2}, w^{i*4+6})`$

## Virtual Subcolumns

Assume now we have 3 columns that share some constraints. For example, let's have three flags that can be either 0 or 1. Each flag will also have its own set of dedicated constraints. 

Let's denote the shared constraint $B$, and the independent constraints $C_{i}$.

What we can do is define a Column for the flags, where the binary constraint $B$ is enforced. Additionally, we will define a subcolumn for each flag, which will enforce each $C_{i}$.

In summary, if we have a set of shared constraints to apply, we will be using a Column. If we want to mix or interleave Columns, we will define them as Virtual Columns. And if we want to apply more constraints to a subset of a Column of Virtual Columns, or share constraints between columns, we will define Virtual Subcolumns.
