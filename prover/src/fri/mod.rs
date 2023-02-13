pub fn hello() {
    println!("Hello world");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_hello() {
        super::hello();
    }
}
