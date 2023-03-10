use crate::FE;

pub struct TraceTable {
    table: Vec<Vec<FE>>,
}

impl TraceTable {
    fn new(rows: Vec<Vec<FE>>) -> Self {
        Self { table: rows }
    }
}
