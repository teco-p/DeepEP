#pragma once
#include <sdaa_runtime.h>
#include <cstdint>
extern "C" sdaaError_t deep_ep_cast_indices(sdaaStream_t, const void*, int, int*, int, int, int64_t*);
extern "C" sdaaError_t deep_ep_expand_dispatch(sdaaStream_t, const void*, const int*,
 const float*, const int*, void*, float*, int*, int*, int*, int*, int*, int*,
 int, int, int, int, int, int, int64_t*);
extern "C" sdaaError_t deep_ep_combine_expanded(sdaaStream_t, const void*, const int*,
 void*, int, int, int, int64_t*);
