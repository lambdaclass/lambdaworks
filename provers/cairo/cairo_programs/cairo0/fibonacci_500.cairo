func main() {
    // Call fib(1, 1, 500).
    let result: felt = fib(1, 1, 500);

    // Make sure the 500th Fibonacci number is 2703462216091053821850160095716009632185810905688261857143077152353261240886.
    assert result = 2703462216091053821850160095716009632185810905688261857143077152353261240886; 
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

