#include "routing.hpp"
#include "runtime_check.hpp"
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_sdaa/sdaa_extension.h>
#include <pybind11/pybind11.h>

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
    m.def("cast_indices", [](const at::Tensor& in, at::Tensor out, at::Tensor status, int experts) {
        TORCH_CHECK(in.is_contiguous() && (in.scalar_type() == at::kInt || in.scalar_type() == at::kLong),
                    "top-k indices must be contiguous int32/int64");
        check_tensor(out, at::kInt, in); check_status(status, in);
        TORCH_CHECK(out.sizes() == in.sizes() && experts == 384, "top-k shape/experts mismatch");
        ep_sdaa_check(deep_ep_cast_indices(torch::sdaa::getCurrentSDAAStream(-1), in.const_data_ptr(),
            in.element_size() * 8, out.data_ptr<int>(), in.numel(), experts, status.data_ptr<int64_t>()),
            "deep_ep_cast_indices", "cast_indices", __LINE__);
    });
    m.def("compact_dispatch", [](const at::Tensor& hidden, const at::Tensor& ids,
        const at::Tensor& weights, const at::Tensor& counts, at::Tensor out,
        at::Tensor out_ids, at::Tensor out_weights, at::Tensor ranks, at::Tensor experts,
        at::Tensor inverse, at::Tensor source, int rank, at::Tensor status) {
        const auto capacity = out.size(0), width = out.size(1);
        check_tensor(hidden, at::kHalf, out); check_tensor(out, at::kHalf, out);
        check_tensor(ids, at::kInt, out); check_tensor(weights, at::kFloat, out);
        for (const auto& t : {counts, out_ids, ranks, experts, inverse, source}) check_tensor(t, at::kInt, out);
        check_tensor(out_weights, at::kFloat, out); check_status(status, out);
        TORCH_CHECK(width == 5120 && hidden.sizes() == at::IntArrayRef({capacity, width + 16})
            && ids.sizes() == at::IntArrayRef({capacity, 6}) && weights.sizes() == ids.sizes()
            && out_ids.sizes() == ids.sizes() && out_weights.sizes() == ids.sizes()
            && counts.numel() == 5 && ranks.numel() == 16 && experts.numel() == 24
            && inverse.numel() == capacity && source.sizes() == at::IntArrayRef({capacity, 2})
            && rank >= 0 && rank < 16, "DeepEP compact shape/topology mismatch");
        ep_sdaa_check(deep_ep_compact(torch::sdaa::getCurrentSDAAStream(-1), hidden.const_data_ptr(),
            ids.const_data_ptr<int>(), weights.const_data_ptr<float>(), counts.const_data_ptr<int>(),
            out.data_ptr(), out_ids.data_ptr<int>(), out_weights.data_ptr<float>(), ranks.data_ptr<int>(),
            experts.data_ptr<int>(), inverse.data_ptr<int>(), source.data_ptr<int>(), capacity, width,
            rank, status.data_ptr<int64_t>()), "deep_ep_compact", "compact_dispatch", __LINE__);
    });
    m.def("scatter_combine", [](const at::Tensor& input, const at::Tensor& inverse,
        at::Tensor output, at::Tensor status) {
        check_tensor(input, at::kHalf, output); check_tensor(output, at::kHalf, output);
        check_tensor(inverse, at::kInt, input); check_status(status, input);
        TORCH_CHECK(input.sizes() == output.sizes() && input.dim() == 2 && input.size(1) == 5120
            && inverse.numel() == input.size(0), "DeepEP inverse/output shape mismatch");
        ep_sdaa_check(deep_ep_scatter(torch::sdaa::getCurrentSDAAStream(-1), input.const_data_ptr(),
            inverse.const_data_ptr<int>(), output.data_ptr(), input.size(0), input.size(1),
            status.data_ptr<int64_t>()), "deep_ep_scatter", "scatter_combine", __LINE__);
    });
}
