%builtins range_check

from starkware.cairo.common.math import assert_nn

func main{range_check_ptr: felt}() {
    assert_nn(5);
    assert_nn(2);
    return ();
}
