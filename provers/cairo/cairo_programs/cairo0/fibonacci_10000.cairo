func main() {
    // Call fib(1, 1, 10000).
    let result: felt = fib(1, 1, 10000);

    // Make sure the 10000th Fibonacci number is 2287375788429092341882876480321135809824733217263858843173749298459021701670.
    assert result = 2287375788429092341882876480321135809824733217263858843173749298459021701670; 
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

