// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu

#include <string>
#include <torch/extension.h>
#include <vector>

/*
 * MUSA forward declarations
 */

std::vector<torch::Tensor> render_forward_musa(torch::Tensor sigma,
                                               torch::Tensor origin,
                                               torch::Tensor points,
                                               torch::Tensor tindex,
                                               const std::vector<int> grid,
                                               std::string phase_name);

std::vector<torch::Tensor>
render_musa(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
            torch::Tensor tindex, std::string loss_name);

torch::Tensor init_musa(torch::Tensor points, torch::Tensor tindex,
                        const std::vector<int> grid);


/*
 * C++ interface
 */

#define CHECK_MUSA(x)                                                          \
  TORCH_CHECK(x.device().str() == "musa", #x " must be a MUSA tensor")         
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_MUSA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
render_forward(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
               torch::Tensor tindex, const std::vector<int> grid,
               std::string phase_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return render_forward_musa(sigma, origin, points, tindex, grid, phase_name);
}


std::vector<torch::Tensor> render(torch::Tensor sigma, torch::Tensor origin,
                                  torch::Tensor points, torch::Tensor tindex,
                                  std::string loss_name) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return render_musa(sigma, origin, points, tindex, loss_name);
}

torch::Tensor init(torch::Tensor points, torch::Tensor tindex,
                   const std::vector<int> grid) {
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return init_musa(points, tindex, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "Initialize");
  m.def("render", &render, "Render");
  m.def("render_forward", &render_forward, "Render (forward pass only)");
}
