%builtins range_check

from starkware.cairo.common.math_cmp import is_le

func main{range_check_ptr: felt}() {
    let result = is_le(2, 6);
    assert result = 1;
    return ();
}

