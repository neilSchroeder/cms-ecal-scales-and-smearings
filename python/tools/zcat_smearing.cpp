#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cmath>

namespace py = pybind11;

py::array_t<float> apply_smearing(py::array_t<float> mc, float lead_smear, float sublead_smear, uint64_t seed) {
    auto buf = mc.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input array must be one-dimensional");

    size_t N = buf.shape[0];
    float* mc_ptr = static_cast<float*>(buf.ptr);

    // Create an output array with the same shape
    auto result = py::array_t<float>(buf.shape);
    auto res_buf = result.request();
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist1(1.0f, lead_smear);
    std::normal_distribution<float> dist2(1.0f, sublead_smear);

    for (size_t i = 0; i < N; i++) {
        float r1 = dist1(rng);
        float r2 = dist2(rng);
        res_ptr[i] = mc_ptr[i] * std::sqrt(r1 * r2);
    }
    return result;
}

PYBIND11_MODULE(csmearing, m) {
    m.doc() = "C++ extension for smearing function";
    m.def("apply_smearing", &apply_smearing,
          "Apply smearing to a 1D numpy array of floats",
          py::arg("mc"), py::arg("lead_smear"), py::arg("sublead_smear"), py::arg("seed"));
}