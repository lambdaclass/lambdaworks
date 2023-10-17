%builtins output

from starkware.cairo.common.serialize import serialize_word

func main{output_ptr: felt*}() {
    const MY_INT = 1234;

    // this will print MY_INT
    serialize_word(MY_INT);

    return ();
}
