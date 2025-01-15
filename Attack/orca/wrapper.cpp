#include "./ORCARunningAlgorithm.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::shared_ptr<ORCARunningAlgorithm> init_orca() {
  return std::make_shared<ORCARunningAlgorithm>();
}

void calculate_next_positions_wrapper(
    double time_step, std::vector<std::pair<double, double>> &positions,
    std::vector<std::pair<double, double>> &velocities, int spoofed_agent_index,
    std::pair<double, double> spoofed_agent_position,
    std::shared_ptr<ORCARunningAlgorithm> orca_instance) {

  auto cpp_positions = std::make_shared<std::vector<Eigen::Vector2d>>();
  for (const auto &p : positions) {
    cpp_positions->emplace_back(p.first, p.second);
  }

  auto cpp_velocities = std::make_shared<std::vector<Eigen::Vector2d>>();
  for (const auto &v : velocities) {
    cpp_velocities->emplace_back(v.first, v.second);
  }

  orca_instance->calculate_next_positions(
      std::make_shared<double>(time_step), cpp_positions, cpp_velocities,
      spoofed_agent_index,
      Eigen::Vector2d(spoofed_agent_position.first,
                      spoofed_agent_position.second));

  // the call was call-by-ref, so we need to copy the results back to our lists.
  for (size_t i = 0; i < cpp_velocities->size(); ++i) {
    velocities[i] = {(*cpp_velocities)[i].x(), (*cpp_velocities)[i].y()};
  }
  for (size_t i = 0; i < cpp_positions->size(); ++i) {
    positions[i] = {(*cpp_positions)[i].x(), (*cpp_positions)[i].y()};
  }
}

void set_initial_positions_from_file_wrapper(
    std::shared_ptr<ORCARunningAlgorithm> orca_instance,
    const std::string &file_path) {
  orca_instance->set_initial_positions_from_file(file_path);
}

std::vector<std::pair<double, double>>
get_positions_wrapper(std::shared_ptr<ORCARunningAlgorithm> orca_instance) {
  auto cpp_positions = orca_instance->get_positions();
  std::vector<std::pair<double, double>> positions;
  for (const auto &p : *cpp_positions) {
    positions.push_back({p.x(), p.y()});
  }
  return positions;
}

std::vector<std::pair<double, double>>
get_velocities_wrapper(std::shared_ptr<ORCARunningAlgorithm> orca_instance) {
  auto cpp_velocities = orca_instance->get_velocities();
  std::vector<std::pair<double, double>> velocities;
  for (const auto &v : *cpp_velocities) {
    velocities.push_back({v.x(), v.y()});
  }
  return velocities;
}

PYBIND11_MODULE(orca_module, m) {
  m.def("init_orca", &init_orca);
  m.def("calculate_next_positions", &calculate_next_positions_wrapper,
        py::arg("time_step"), py::arg("positions"), py::arg("velocities"),
        py::arg("spoofed_agent_index"), py::arg("spoofed_agent_position"),
        py::arg("orca_instance"));
  m.def("get_positions", &get_positions_wrapper, py::arg("orca_instance"));
  m.def("get_velocities", &get_velocities_wrapper, py::arg("orca_instance"));

  py::class_<ORCARunningAlgorithm, std::shared_ptr<ORCARunningAlgorithm>>(
      m, "ORCARunningAlgorithm")
      .def(py::init<>())
      .def("set_initial_positions_from_file",
           &ORCARunningAlgorithm::set_initial_positions_from_file)
      .def("set_initial_velocities",
           &ORCARunningAlgorithm::set_initial_velocities);
}
