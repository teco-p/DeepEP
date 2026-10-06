#include "routing.hpp"
#include "runtime_check.hpp"
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_sdaa/sdaa_extension.h>
#include <pybind11/pybind11.h>
#include <limits>

namespace {
void check_tensor(const at::Tensor& t, at::ScalarType type, const at::Tensor& reference) {
    TORCH_CHECK(t.device() == reference.device() && t.is_contiguous() && t.scalar_type() == type,
                "DeepEP SDAA routing dtype/device/stride mismatch");
}
void check_status(const at::Tensor& t, const at::Tensor& reference) {
    check_tensor(t, at::kLong, reference);
    TORCH_CHECK(t.numel() == 4, "DeepEP status must be four device int64 words");
}
}
void bind_sdaa_routing(pybind11::module_& m) {
    m.def("expand_dispatch", [](const at::Tensor& hidden, const at::Tensor& ids,
        const at::Tensor& weights, const at::Tensor& counts, at::Tensor output,
        at::Tensor out_weights, at::Tensor ranks, at::Tensor experts,
        at::Tensor expert_counts, at::Tensor padded_counts, at::Tensor route_slots,
        at::Tensor source, int rank, at::Tensor status, int alignment) {
        TORCH_CHECK(hidden.dim() == 2 && output.dim() == 2 &&
            output.device().type() == at::kPrivateUse1,
            "DeepEP expanded hidden/output must be SDAA matrices");
        const auto capacity = hidden.size(0), width = output.size(1);
        const auto expanded_capacity = output.size(0);
        check_tensor(hidden, at::kHalf, output); check_tensor(output, at::kHalf, output);
        check_tensor(ids, at::kInt, output); check_tensor(weights, at::kFloat, output);
        check_tensor(out_weights, at::kFloat, output); check_status(status, output);
        for (const auto& tensor : {counts, ranks, experts, expert_counts, padded_counts,
                                  route_slots, source}) check_tensor(tensor, at::kInt, output);
        TORCH_CHECK(capacity > 0 && capacity % 4 == 0 &&
            capacity <= std::numeric_limits<int>::max() / 6 &&
            expanded_capacity <= std::numeric_limits<int>::max() &&
            alignment > 0 && (alignment & (alignment - 1)) == 0 &&
            expanded_capacity == capacity * 6 + 24LL * (alignment - 1) &&
            width == 5120 && hidden.size(1) == width + 16 &&
            ids.sizes() == at::IntArrayRef({capacity, 6}) && weights.sizes() == ids.sizes() &&
            counts.sizes() == at::IntArrayRef({5}) && ranks.sizes() == at::IntArrayRef({16}) &&
            experts.sizes() == at::IntArrayRef({24}) &&
            expert_counts.sizes() == at::IntArrayRef({24}) &&
            padded_counts.sizes() == at::IntArrayRef({24}) &&
            out_weights.sizes() == at::IntArrayRef({expanded_capacity}) &&
            route_slots.sizes() == at::IntArrayRef({capacity, 6}) &&
            source.sizes() == at::IntArrayRef({capacity, 2}) && rank >= 0 && rank < 16,
            "DeepEP expert-major expanded shape/topology/alignment mismatch");
        ep_sdaa_check(deep_ep_expand_dispatch(torch::sdaa::getCurrentSDAAStream(-1),
            hidden.const_data_ptr(), ids.const_data_ptr<int>(), weights.const_data_ptr<float>(),
            counts.const_data_ptr<int>(), output.data_ptr(), out_weights.data_ptr<float>(),
            ranks.data_ptr<int>(), experts.data_ptr<int>(), expert_counts.data_ptr<int>(),
            padded_counts.data_ptr<int>(), route_slots.data_ptr<int>(), source.data_ptr<int>(),
            capacity, expanded_capacity, width, rank, alignment, status.data_ptr<int64_t>()),
            "deep_ep_expand_dispatch", "expand_dispatch", __LINE__);
    });
    m.def("combine_expanded", [](const at::Tensor& input, const at::Tensor& route_slots,
        at::Tensor output, at::Tensor status) {
        TORCH_CHECK(input.dim() == 2 && output.dim() == 2 &&
            output.device().type() == at::kPrivateUse1,
            "DeepEP expanded combine input/output must be SDAA matrices");
        const auto capacity = output.size(0), expanded_capacity = input.size(0);
        const auto width = output.size(1);
        check_tensor(input, at::kHalf, output); check_tensor(output, at::kHalf, output);
        check_tensor(route_slots, at::kInt, output); check_status(status, output);
        TORCH_CHECK(capacity > 0 && capacity % 4 == 0 &&
            capacity <= std::numeric_limits<int>::max() / 6 &&
            expanded_capacity >= capacity * 6 &&
            expanded_capacity <= std::numeric_limits<int>::max() &&
            width == 5120 && input.size(1) == width &&
            route_slots.sizes() == at::IntArrayRef({capacity, 6}),
            "DeepEP expert-major combine shape mismatch");
        ep_sdaa_check(deep_ep_combine_expanded(torch::sdaa::getCurrentSDAAStream(-1),
            input.const_data_ptr(), route_slots.const_data_ptr<int>(), output.data_ptr(),
            capacity, expanded_capacity, width, status.data_ptr<int64_t>()),
            "deep_ep_combine_expanded", "combine_expanded", __LINE__);
    });
    m.def("cast_indices", [](const at::Tensor& in, at::Tensor out, at::Tensor status, int experts) {
        TORCH_CHECK(in.is_contiguous() && (in.scalar_type() == at::kInt || in.scalar_type() == at::kLong),
                    "top-k indices must be contiguous int32/int64");
        check_tensor(out, at::kInt, in); check_status(status, in);
        TORCH_CHECK(out.sizes() == in.sizes() && experts == 384, "top-k shape/experts mismatch");
        ep_sdaa_check(deep_ep_cast_indices(torch::sdaa::getCurrentSDAAStream(-1), in.const_data_ptr(),
            in.element_size() * 8, out.data_ptr<int>(), in.numel(), experts, status.data_ptr<int64_t>()),
            "deep_ep_cast_indices", "cast_indices", __LINE__);
    });
}
