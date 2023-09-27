use data::pairs;
use solver::solve;

use crate::cypher::evaluate;

mod cypher;
mod data;
mod field;
mod solver;

fn main() {
    let key = solve();

    let (p, c) = pairs()[0].clone();
    assert_eq!(evaluate(&p, &key), c);

    println!("Found Key! {}", &key);
}
