use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Expr {
    Value(String),
    Array(Vec<Expr>),
}

pub trait IntoAst {
    fn into_ast(&self) -> Vec<Expr>;
}