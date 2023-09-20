func main() {
    // Call fib(1, 1, 10).
    let result: felt = fib(1, 1, 100);

    // Make sure the 100th Fibonacci number is 927372692193078999176.
    assert result = 927372692193078999176; 
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

