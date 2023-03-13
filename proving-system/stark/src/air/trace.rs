use crate::FE;

pub struct TraceTable {
    pub table: Vec<Vec<FE>>,
}

impl TraceTable {
    fn new(rows: Vec<Vec<FE>>) -> Self {
        Self { table: rows }
    }
}
