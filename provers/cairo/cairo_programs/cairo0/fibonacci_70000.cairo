func main() {
    // Call fib(1, 1, 70000).
    let result: felt = fib(1, 1, 70000);

    // Make sure the 70000th Fibonacci number is 2824861842081921227084209061704696342102277985526232307654372591609636515753.
    assert result = 2824861842081921227084209061704696342102277985526232307654372591609636515753; 
    ret;
}

func fib(first_element, second_element, n) -> (res: felt) {
    jmp fib_body if n != 0;
    tempvar result = second_element;
    return (second_element,);

    fib_body:
    tempvar y = first_element + second_element;
    return fib(second_element, y, n - 1);
}

