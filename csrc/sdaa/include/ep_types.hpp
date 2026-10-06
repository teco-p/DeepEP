#pragma once

// Recovered from g447683d DWARF, ep_types.hpp:3.
enum EP_DISPATCH : unsigned int {
    EP_RETURN_SUCCESS = 0,
    EP_RETURN_FAILED = 1,
};

// Recovered from g447683d DWARF, ep_types.hpp:8.
enum EP_DTYPE : unsigned int {
    EP_DTYPE_FLOAT16 = 0,
    EP_DTYPE_INT8 = 1,
    EP_DTYPE_FLOAT32 = 2,
    EP_DTYPE_FLOAT64 = 3,
    EP_DTYPE_INT32 = 4,
    EP_DTYPE_INT64 = 5,
};
