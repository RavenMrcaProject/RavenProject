#ifndef ATTACKLIB_ORCARUNNING_ALGORITHM_H
#define ATTACKLIB_ORCARUNNING_ALGORITHM_H

#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include <RVOSimulator.h>
#include <Vector2.h>
#include <attacklib/RunningAlgorithm.h>

/**
 * \file ORCARunningAlgorithm.h
 * @brief A class that implements the ORCA running algorithm.
 *
 */
class ORCARunningAlgorithm : public RunningAlgorithm {
public:
  /**
   * @brief Construct a new ORCARunningAlgorithm object
   *
   */
  void calculate_next_positions(
      std::shared_ptr<double> time_step,
      std::shared_ptr<std::vector<Eigen::Vector2d>> positions,
      std::shared_ptr<std::vector<Eigen::Vector2d>> velocities,
      int spoofed_robot_id,
      Eigen::Vector2d spoofed_drone_real_position) override;
  void calculate_next_positions();
  /**
   * @brief Destroy the ORCARunningAlgorithm object
   *
   */
  ~ORCARunningAlgorithm();
  ORCARunningAlgorithm();
  /**
   * @brief Construct a new ORCARunningAlgorithm object
   *
   */
  ORCARunningAlgorithm(
      int number_of_robots, double time_t, double time_step, double radius,
      std::shared_ptr<std::vector<Eigen::Vector2d>> positions,
      std::shared_ptr<std::vector<Eigen::Vector2d>> goals,
      std::shared_ptr<std::vector<std::vector<Eigen::Vector2d>>> obstacles,
      float neighborDist, int maxNeighbors, float timeHorizon,
      float timeHorizonObst, float maxSpeed);
  void setupScenario(
      double time_step, std::shared_ptr<std::vector<Eigen::Vector2d>> positions,
      std::shared_ptr<std::vector<Eigen::Vector2d>> goals,
      std::shared_ptr<std::vector<std::vector<Eigen::Vector2d>>> obstacles);
  void setPreferredVelocities(std::shared_ptr<RVO::RVOSimulator> sim);
  bool reachedGoal() override;

  void set_goals_from_file(std::string file_path);
  void set_obstacles_from_file(std::string file_path);
  void set_initial_positions_from_file(std::string file_path);

  std::shared_ptr<std::vector<Eigen::Vector2d>> get_positions();
  std::shared_ptr<std::vector<Eigen::Vector2d>> get_velocities();
  Eigen::Vector2d get_velocity(int i);
  Eigen::Vector2d get_position(int i);

  void set_initial_velocities();

private:
  std::shared_ptr<RVO::RVOSimulator> sim;
  std::vector<RVO::Vector2> goals;
  std::vector<RVO::Vector2> obstacles;
  std::vector<RVO::Vector2> positions;
  int number_of_robots;
  double radius;
  double neighborDist;
  int maxNeighbors;
  double timeHorizon;
  double timeHorizonObst;
  double maxSpeed;
  double time_step;
  double time_t;
  std::string environment_file;
  std::ofstream internal_file_for_dataset;

  std::shared_ptr<std::vector<Eigen::Vector2d>> _positions;
  std::shared_ptr<std::vector<Eigen::Vector2d>> _goals;
  std::shared_ptr<std::vector<std::vector<Eigen::Vector2d>>> _obstacles;
  std::shared_ptr<std::vector<Eigen::Vector2d>> _velocities;
};

#endif