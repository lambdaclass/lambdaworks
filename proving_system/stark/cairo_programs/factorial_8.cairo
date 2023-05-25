// factorial(n) =  n!
func factorial(n) -> (result: felt) {
    if (n == 1) {
        return (n,);
    }
    let (a) = factorial(n - 1);
    return (n * a,);
}

func main() {
    // Make sure the factorial(8) == 40320
    let (y) = factorial(8);
    y = 40320;
    return ();
}
