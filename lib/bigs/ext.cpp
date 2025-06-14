/**
 * @file ext.cpp
 * @brief torch extension for featmark
 * @author sailing-innocent
 * @date 2025-02-18
 */

#include <torch/extension.h>
#include "featmark_point.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gs_feat_mark_debug", &sailtorch::gs_feat_mark_debug);
  m.def("gs_feat_mark", &sailtorch::gs_feat_mark);
  m.def("gs_feat_mark_var", &sailtorch::gs_feat_mark_var);
}