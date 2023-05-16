#pragma once

#include "../fields/unsigned_int.h.metal"

namespace {
    typedef UnsignedInteger<12> u384;
}

[[kernel]]
void test_uint_add(
    constant u384& _a [[ buffer(0) ]],
    constant u384& _b [[ buffer(1) ]],
    device u384& result [[ buffer(2) ]])
{
    u384 a = _a;
    u384 b = _b;

    result = a + b;
}

[[kernel]]
void test_uint_sub(
    constant u384& _a [[ buffer(0) ]],
    constant u384& _b [[ buffer(1) ]],
    device u384& result [[ buffer(2) ]])
{
    u384 a = _a;
    u384 b = _b;

    result = a - b;
}

[[kernel]]
void test_uint_prod(
    constant u384& _a [[ buffer(0) ]],
    constant u384& _b [[ buffer(1) ]],
    device u384& result [[ buffer(2) ]])
{
    u384 a = _a;
    u384 b = _b;

    result = a * b;
}

[[kernel]]
void test_uint_shl(
    constant u384& _a [[ buffer(0) ]],
    constant uint64_t& _b [[ buffer(1) ]],
    device u384& result [[ buffer(2) ]])
{
    u384 a = _a;
    uint64_t b = _b;

    result = a << b;
}

[[kernel]]
void test_uint_shr(
    constant u384& _a [[ buffer(0) ]],
    constant uint64_t& _b [[ buffer(1) ]],
    device u384& result [[ buffer(2) ]])
{
    u384 a = _a;
    uint64_t b = _b;

    result = a >> b;
}
