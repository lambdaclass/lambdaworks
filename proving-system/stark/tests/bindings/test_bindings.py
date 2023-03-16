from lambdaworks_stark import *

try:
    add(18446744073709551615, 1)
    assert(not("Should have thrown a IntegerOverflow exception!"))
except ArithmeticError.IntegerOverflow:
    # It's okay!
    pass

assert add(2, 4) == 6
assert add(4, 8) == 12

number = get_number()

assert number.neg() == -2
