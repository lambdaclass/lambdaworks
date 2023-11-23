/// Positions in each trace step of every trace cell

// RC POOL - Column 0
const OFF_DST: (usize, usize) = (0, 0);
const OFF_OP0: (usize, usize) = (0, 4);
const OFF_OP1: (usize, usize) = (0, 8);

// FLAGS - Column 1
const WHOLE_FLAG_PREFIX: (usize, usize) = (1, 0);

// MEM POOL - Column 3
const INSTR_ADDR: (usize, usize) = (3, 0);
const INSTR_VALUE: (usize, usize) = (3, 1);
const PUB_MEM_ADDR: (usize, usize) = (3, 2);
const PUB_MEM_VALUE: (usize, usize) = (3, 3);
const OP0_ADDR: (usize, usize) = (3, 4);
const OP0_VALUE: (usize, usize) = (3, 5);
const DST_ADDR: (usize, usize) = (3, 8);
const DST_VALUE: (usize, usize) = (3, 9);
const OP1_ADDR: (usize, usize) = (3, 12);
const OP1_VALUE: (usize, usize) = (3, 13);

// PC UPDATE - Column 5
const AP: (usize, usize) = (5, 0);
const TMP0: (usize, usize) = (5, 2);
const OPS_MUL: (usize, usize) = (5, 4);
const FP: (usize, usize) = (5, 8);
const TMP1: (usize, usize) = (5, 10);
const RES: (usize, usize) = (5, 12);
