func mul(x: felt, y: felt) -> (res: felt) {
    return (res = x * y);
}

func main() {
    let x = 2;
    let y = 3;

    let (res) = mul(x, y);
    assert res = 6;

    return ();
}
