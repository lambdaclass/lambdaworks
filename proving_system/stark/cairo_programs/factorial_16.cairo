// factorial(n) =  n!
func factorial(n) -> (result: felt) {
    if (n == 1) {
        return (n,);
    }
    let (a) = factorial(n - 1);
    return (n * a,);
}

func main() {
    // Make sure the factorial(16) == 20922789888000
    let (y) = factorial(16);
    y = 20922789888000;
    return ();
}
