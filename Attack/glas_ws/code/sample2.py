import logging
import math
import os
from typing import Any, List
import numpy as np
# import plotly.graph_objects as go
from staliro.core.interval import Interval
from staliro.core import best_eval, best_run
from staliro.core.model import FailureResult
from staliro.core.result import worst_eval, worst_run
from staliro.models import State, ode, blackbox, BasicResult, Model, ModelInputs, ModelResult, Trace
from staliro.optimizers import UniformRandom, DualAnnealing
from staliro.options import Options
from staliro.specifications import RTAMTDense, RTAMTDiscrete
from staliro.staliro import simulate_model, staliro
import math
import random
# from collision_avoidance import CollisionAvoidanceSystem

import os
from examples import run_singleintegrator
from systems.singleintegrator import SingleIntegrator
from examples.run import run, parse_args
import torch
import numpy as np
from collections import namedtuple
from sim import run_sim
import yaml
import plotter
from matplotlib.patches import Rectangle, Circle

from examples.run_singleintegrator import SingleIntegratorParam
import pickle
from decimal import Decimal, ROUND_HALF_UP, localcontext
import shutil


def point_line_distance(point, line_start, line_end):
    """Calculate distance from a point to a line segment."""
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    line_vec = line_end - line_start
    point_vec = point - line_start
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        return np.linalg.norm(point_vec)

    t = max(0, min(1, np.dot(point_vec, line_vec) / (line_length * line_length)))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def get_square_corners(bottom_left):
    """Convert bottom-left corner point into four corners of 1x1 square."""
    x, y = bottom_left
    return [
        (x, y),        # bottom-left
        (x + 1, y),    # bottom-right
        (x + 1, y + 1),  # top-right
        (x, y + 1)     # top-left
    ]


def calculate_min_distance_to_obstacles(point, obstacles):
    """Calculate minimum distance from a point to any obstacle."""
    min_dist = float('inf')

    for obstacle in obstacles:
        corners = get_square_corners(obstacle)
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            dist = point_line_distance(point, p1, p2)
            min_dist = min(min_dist, dist)

    return min_dist


class CollisionAvoidanceSystem:
    def __init__(self):
        # initiate glas

        # parse the config file
        # variables = self.parse_config(
        #     "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")
        # self.numRobots = int(variables['num_robots'])
        # self.timeStep = float(variables['time_step'])
        # self.environmentFile = variables['environment_file']
        # self.radius = float(variables['radius'])

        self.args = parse_args()
        # print("here", self.args)
        self.param = SingleIntegratorParam()
        # print("here", self.param)
        # print("heyo", self.param.n_agents)

        self.env = SingleIntegrator(self.param)
        # print("here", self.env)

        # print(self.param.il_train_model_fn)
        print("here ", self.param.il_train_model_fn)
        self.controllers = {
            # Use custom load function
            'current': torch.load(self.param.il_train_model_fn),
        }

        self.s0 = run_singleintegrator.load_instance(
            self.param, self.env, self.args.instance)

        # print("heyo3", self.param.n_agents)

        self.observations = []
        self.reward = 0

        self.states = np.empty((len(self.param.sim_times), self.env.n))
        self.actions = np.empty((len(self.param.sim_times)-1, self.env.m))

        self.env.reset(self.s0)
        self.states[0] = np.copy(self.env.s)

        self.done = False
        self.step = 0

        self.robot_initial_positions = []
        self.goal_positions = []
        self.obstacles = []

        torch.set_num_threads(1)
        if self.s0 is None:
            self.s0 = self.env.reset()

        self.sim_results = []

        # print("heyo22", self.env.agents)
        # print(self.env.agents)

    def parse_config(self, file_path):
        variables = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    key_value = line.split('=')
                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        value = value.strip().strip('"')
                        variables[key] = value
        return variables

    def reached_goal(self):
        # check if the agents have reached the goal
        return self.env.done()

    def set_agent_position(self, agent_id, x, y):
        self.env.update_agent_pos(agent_id, x, y)

    # TODO: set_vel may not be working
    def set_agent_velocity(self, agent_id, vx, vy):
        self.env.update_agent_vel(agent_id, vx, vy)

    def get_agent_positions(self):
        # get the agent positions
        positions = []
        for agent in self.env.agents:
            positions.append(agent.p)
        return positions

    def get_agent_velocities(self):
        # get the agent velocities
        velocities = []
        for agent in self.env.agents:
            velocities.append(agent.v)
        return velocities

    def print_agent_positions(self):
        # print the agent positions
        print("Agent positions")
        for agent_position in self.get_agent_positions():
            print(agent_position[0])
            print(agent_position[1])

    def print_agent_velocities(self):
        print("Agent velocities")
        for agent_velocity in self.get_agent_velocities():
            print(agent_velocity)

    def calculate_next_velocities_(self, agentID, agentPos):
        state = self.states[self.step]
        observation = self.env.observe()

        for name, controller in self.controllers.items():
            action = controller.policy(observation)
            next_state, r, done, _ = self.env.step_(
                action, False, agentID, agentPos)
            self.reward += r

            self.states[self.step + 1] = next_state
            self.actions[self.step] = action.flatten()
            self.observations.append(observation)

            self.step += 1
            self.done = done
        return self.states, self.observations, self.actions, self.step

    def calculate_next_velocities(self,):
        # calculate the next velocities of the agents
        state = self.states[self.step]
        observation = self.env.observe()

        for name, controller in self.controllers.items():
            action = controller.policy(observation)
            next_state, r, done, _ = self.env.step(
                action, compute_reward=False)
            self.reward += r

            self.states[self.step + 1] = next_state
            self.actions[self.step] = action.flatten()
            self.observations.append(observation)

            self.step += 1
            self.done = done
        return self.states, self.observations, self.actions, self.step

    def pack_results(self):
        # states = np.empty((len(self.param.sim_times), self.env.n))
        # states[0] = np.copy(self.env.s)
        # actions = np.empty((len(self.param.sim_times)-1, self.env.m))
        # observations = []
        # reward = 0
        # controller_name = None
        # for name, controller in self.controllers.items():
        #     for i in range(len(self.sim_results)):
        #         states[i+1] = self.sim_results[i].states
        #         actions[i] = self.sim_results[i].actions
        #         observations.append(self.sim_results[i].observations)
        #         reward += self.sim_results[i].reward
        #         controller_name = name

        # return SimResult._make(self.states, self.observations, self.actions, self.step + (controller_name, ))
        big_list = []
        for result in self.sim_results:
            big_list.append(result)

    def draw(self):
        # plot state space
        times = self.param.sim_times
        result = self.pack_results()
        if self.param.env_name in ['SingleIntegrator', 'SingleIntegratorVelSensing', 'DoubleIntegrator']:
            fig, ax = plotter.make_fig()
            ax.set_title('State Space')
            ax.set_aspect('equal')

            for o in self.env.obstacles:
                ax.add_patch(
                    Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

            for agent in self.env.agents:

                line = ax.plot(result.states[0:result.steps, self.env.agent_idx_to_state_idx(agent.i)],
                               result.states[0:result.steps, self.env.agent_idx_to_state_idx(agent.i)+1], alpha=0.5)
                color = line[0].get_color()

                # plot velocity vectors:
                X = []
                Y = []
                U = []
                V = []
                for k in np.arange(0, result.steps, 100):
                    X.append(
                        result.states[k, self.env.agent_idx_to_state_idx(agent.i)])
                    Y.append(
                        result.states[k, self.env.agent_idx_to_state_idx(agent.i)+1])
                    if self.param.env_name in ['SingleIntegrator', 'SingleIntegratorVelSensing']:
                        # Singleintegrator: plot actions
                        U.append(result.actions[k, 2*agent.i+0])
                        V.append(result.actions[k, 2*agent.i+1])
                    elif self.param.env_name in ['DoubleIntegrator']:
                        # doubleintegrator: plot velocities
                        U.append(
                            result.states[k, self.env.agent_idx_to_state_idx(agent.i)+2])
                        V.append(
                            result.states[k, self.env.agent_idx_to_state_idx(agent.i)+3])

                ax.quiver(X, Y, U, V, angles='xy', scale_units='xy',
                          scale=0.5, color=color, width=0.005)
                plotter.plot_circle(result.states[1, self.env.agent_idx_to_state_idx(agent.i)],
                                    result.states[1, self.env.agent_idx_to_state_idx(agent.i)+1], self.param.r_agent, fig=fig, ax=ax, color=color)
                plotter.plot_square(
                    agent.s_g[0], agent.s_g[1], self.param.r_agent, angle=45, fig=fig, ax=ax, color=color)

            # draw state for each time step
            robot = 0
            if self.param.env_name in ['SingleIntegrator']:
                for step in np.arange(0, result.steps, 1000):
                    fig, ax = plotter.make_fig()
                    ax.set_title('State at t={} for robot={}'.format(
                        times[step], robot))
                    ax.set_aspect('equal')

                    # plot all obstacles
                    for o in self.env.obstacles:
                        ax.add_patch(
                            Rectangle(o, 1.0, 1.0, facecolor='gray', alpha=0.5))

                    # plot overall trajectory
                    line = ax.plot(result.states[0:result.steps, self.env.agent_idx_to_state_idx(robot)],
                                   result.states[0:result.steps, self.env.agent_idx_to_state_idx(robot)+1], "--")
                    color = line[0].get_color()

                    # plot current position
                    plotter.plot_circle(result.states[step, self.env.agent_idx_to_state_idx(robot)],
                                        result.states[step, self.env.agent_idx_to_state_idx(robot)+1], self.param.r_agent, fig=fig, ax=ax, color=color)

                    # plot current observation
                    observation = result.observations[step][robot][0]
                    num_neighbors = int(observation[0])
                    num_obstacles = int(
                        (observation.shape[0]-3 - 2*num_neighbors)/2)

                    robot_pos = result.states[step, self.env.agent_idx_to_state_idx(
                        robot):self.env.agent_idx_to_state_idx(robot)+2]

                    idx = 3
                    for i in range(num_neighbors):
                        pos = observation[idx: idx+2] + robot_pos
                        ax.add_patch(
                            Circle(pos, 0.25, facecolor='gray', edgecolor='red', alpha=0.5))
                        idx += 2

                    for i in range(num_obstacles):
                        # pos = observation[idx : idx+2] + robot_pos - np.array([0.5,0.5])
                        # ax.add_patch(Rectangle(pos, 1.0, 1.0, facecolor='gray', edgecolor='red', alpha=0.5))
                        pos = observation[idx: idx+2] + robot_pos
                        ax.add_patch(
                            Circle(pos, 0.5, facecolor='gray', edgecolor='red', alpha=0.5))
                        idx += 2

                    # plot goal
                    goal = observation[1:3] + robot_pos
                    ax.add_patch(
                        Rectangle(goal - np.array([0.2, 0.2]), 0.4, 0.4, alpha=0.5, color=color))

                # 	# import matplotlib.pyplot as plt
                # 	# plt.savefig("test.svg")
                # 	# exit()

        # plot time varying states
        if self.param.env_name in ['SingleIntegrator', 'DoubleIntegrator']:
            for i_config in range(self.env.state_dim_per_agent):
                fig, ax = plotter.make_fig()
                ax.set_title(self.env.states_name[i_config])
                for agent in self.env.agents:
                    for result in self.sim_results:
                        ax.plot(
                            times[1:result.steps],
                            result.states[1:result.steps, self.env.agent_idx_to_state_idx(
                                agent.i)+i_config],
                            label=result.name)

        # plot time varying actions
        if self.param.env_name in ['SingleIntegrator', 'DoubleIntegrator']:
            for i_config in range(self.env.action_dim_per_agent):
                fig, ax = plotter.make_fig()
                ax.set_title(self.env.actions_name[i_config])
                for agent in self.env.agents:
                    for result in self.sim_results:
                        ax.plot(
                            times[1:result.steps],
                            result.actions[1:result.steps, agent.i *
                                           self.env.action_dim_per_agent+i_config],
                            label=result.name)

                        #
                        if i_config == 5:
                            ax.set_yscale('log')

        plotter.save_figs(self.param.plots_fn)
        plotter.open_figs(self.param.plots_fn)

        # visualize
        self.env.visualize(self.sim_results[0].states[0:result.steps], 0.1)

    def write_positions_to_file(self, time_step):
        # write the agent positions to a file
        with open('agent_positions.txt', 'a') as f:
            f.write(str(time_step) + " ")
            agent_pos_list = self.get_agent_positions()
            for i in agent_pos_list:
                # write with .2f precision
                f.write("({:.4f},{:.4f}) ".format(
                    i[0], i[1]))

            f.write("\n")

    def delete_file(self, ):
        # delete the file if exists
        if os.path.exists("agent_positions.txt"):
            os.remove("agent_positions.txt")


def parse_config(filename):
    config = {}
    with open(filename, 'r') as file:
        key, value = None, ""
        for line in file:
            # Remove whitespace and skip comments
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Check if line continues a previous value
            if line.startswith('"') and key is not None:
                value += line.strip('"')
                config[key] = value
                key, value = None, ""
                continue

            # Split on the first equals sign
            parts = line.split('=', 1)
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue

            key, value = parts[0].strip(), parts[1].strip().strip('"')

            # Check for multi-line values
            if value.endswith("\\"):
                value = value[:-1].strip('"')
            else:
                config[key] = value
                key, value = None, ""

    return config


algorithm = "glas_january"
# TODO: change the path to the config file
config = parse_config(
    "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")
num_of_robots = int(config.get("num_robots", 0))
radius = float(config.get("radius", 0.0))
time_step = float(config.get("time_step", 0.0))
environment_file = config.get("environment_file", "")
environment_name = os.path.basename(os.path.normpath(environment_file))
yaml_name = os.path.basename(
    os.path.normpath(environment_file)).replace("/", "")
maxSpeed = float(config.get("maxSpeed", 0.0))
minPosChange = float(config.get("minPosChange", 0.0))
maxPosChange = float(config.get("maxPosChange", 0.0))
minVelChange = float(config.get("minVelChange", 0.0))
maxVelChange = float(config.get("maxVelChange", 0.0))
numberOfFalseMessage = int(config.get("numberOfFalseMessage", 0))
attackType = int(config.get("attackType", 0))
victimRobotId = int(config.get("victimRobotId", 0))
attackedRobotId = int(config.get("attackedRobotId", 0))
deadlockTimestep = int(config.get("deadlockTimestep", 0))
deadlockPosChange = float(config.get("deadlockPosChange", 0.0))
totalTimeStep = float(config.get("totalTimeStep", 0.0))
falsificationIterations = int(config.get("falsificationIterations", 0))
falsificationRuns = int(config.get("falsificationRuns", 0))
delayConstant = int(config.get("delayConstant", 0))
pointX = float(config.get("pointX", 0))
pointY = float(config.get("pointY", 0))


def print_all_config_params():
    print("num_of_robots: ", num_of_robots)
    print("radius: ", radius)
    print("time_step: ", time_step)
    print("environment_name: ", environment_name)
    print("maxSpeed: ", maxSpeed)
    print("numberOfFalseMessage: ", numberOfFalseMessage)
    print("attackType: ", attackType)
    print("victimRobotId: ", victimRobotId)
    print("attackedRobotId: ", attackedRobotId)
    print("deadlockTimestep: ", deadlockTimestep)
    print("deadlockPosChange: ", deadlockPosChange)
    print("totalTimeStep: ", totalTimeStep)
    print("falsificationIterations: ", falsificationIterations)
    print("falsificationRuns: ", falsificationRuns)
    print("delayConstant: ", delayConstant)
    print("pointX: ", pointX)
    print("pointY: ", pointY)


def process_initial_conditions(initial_conditions):
    # print("Initial conditions: ", initial_conditions)
    result = {}
    for i in range(0, len(initial_conditions), 3):
        key = round_to_nearest_step(initial_conditions[i], step=time_step)
        values = tuple(initial_conditions[i+1:i+3])
        result[key] = values
    return result


def round_to_nearest_step(number, step=0.25):
    number = Decimal(str(number))
    step = Decimal(str(step))
    rounded_value = (number / step).quantize(Decimal('1'),
                                             rounding=ROUND_HALF_UP) * step
    return float(rounded_value)


def increment_precise(number, increment, precision=10):
    with localcontext() as ctx:
        ctx.prec = precision
        num_decimal = Decimal(str(number))
        inc_decimal = Decimal(str(increment))
        result = num_decimal + inc_decimal
        result = result.quantize(Decimal('0.0'), rounding=ROUND_HALF_UP)
    return float(result)


def calculate_abs_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def load_obstacles_from_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    obstacles = data.get('map', {}).get('obstacles', [])
    return obstacles


def load_goals_from_yaml(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)

    goals = [agent.get('goal') for agent in data.get('agents', [])]
    return goals

    # Calculate the minimum distance between a square obstacle and a circular drone.
    square_x_min, square_y_min = corner_point
    square_x_max = square_x_min + 1 if square_x_min >= 0 else square_x_min - 1
    square_y_max = square_y_min + 1 if square_y_min >= 0 else square_y_min - 1

    circle_x, circle_y = circle_center

    # Calculate the closest point on the square to the circle center
    closest_x = np.clip(circle_x, min(
        square_x_min, square_x_max), max(square_x_min, square_x_max))
    closest_y = np.clip(circle_y, min(
        square_y_min, square_y_max), max(square_y_min, square_y_max))

    # Calculate the distance between the circle center and the closest point
    distance = np.sqrt((closest_x - circle_x) ** 2 +
                       (closest_y - circle_y) ** 2)

    # Subtract the circle's radius to get the distance from the edge of the circle
    # distance -= circle_radius

    # Ensure distance is non-negative (when the circle intersects the square)
    distance = max(0, distance)

    return distance


def parse_obstacle_centers_glas(file_path):
    """Parse obstacles from YAML file. Returns list of bottom-left corners."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    obstacles = data.get('map', {}).get('obstacles', [])
    return obstacles  # Return corners directly, no conversion needed


# TODO: change the path to the config file
obstacles = parse_obstacle_centers_glas(
    "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
print("Obstacle bottom left corners: ", obstacles)

# TODO: change the path to the config file
goal_positions = load_goals_from_yaml(
    "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
print("Goal positions: ", goal_positions)

# create log of robot positions for last falsification
global_log_list_last = []
global_log_list_prev = []
global_log_list_fake = []
global_log_list_last_with_velocities = []

# create log of robot positions
global_log_list = []

global_min_dist_of_victim_to_static_so_far = float('inf')
global_min_dist_of_any_to_static_so_far = float('inf')
global_min_dist_of_victim_to_robots_so_far = float('inf')
global_min_dist_of_any_to_robots_so_far = float('inf')
global_min_navigating_to_point_victim = float('inf')
global_min_navigating_to_point_any = float('inf')
global_min_deadlock_of_victim_robot = float('inf')
global_min_deadlock_of_any_robot = float('inf')
global_min_navigation_duration_of_victim_robot = float('inf')
global_max_navigation_duration_of_any_robot = -float('inf')

global_iteration = 0


@blackbox()
def Glas_Model(state: State, time: Interval, _: Any) -> BasicResult:
    global global_iteration
    global_iteration += 1

    global global_log_list_last
    global global_log_list_prev
    global global_log_list_fake
    global global_log_list_last_with_velocities

    global global_log_list
    global global_min_dist_of_victim_to_static_so_far
    global global_min_dist_of_any_to_static_so_far
    global global_min_dist_of_victim_to_robots_so_far
    global global_min_dist_of_any_to_robots_so_far
    global global_min_navigating_to_point_victim
    global global_min_navigating_to_point_any
    global global_min_deadlock_of_victim_robot
    global global_min_deadlock_of_any_robot
    global global_min_navigation_duration_of_victim_robot
    global global_max_navigation_duration_of_any_robot

    counter = 0.0

    time_result = []
    list_min_obstacle_dist_of_victim = []
    list_min_obstacle_dist_of_any_robot = []
    list_min_inter_robot_dist_of_victim_robot = []
    list_min_inter_robot_dist_of_any_robot = []
    list_min_deadlock_of_victim_robot = []
    list_min_deadlock_of_any_robot = []
    list_min_navigation_duration_of_victim_robot = []
    list_max_navigation_duration_of_any_robot = []
    list_min_navigating_to_point_victim = []
    list_min_navigating_to_point_any = []
    list_min_deadlock_for_navigation_duration_of_any_robot = []

    last_positions = []
    last_positions_dict = {robot_id: [] for robot_id in range(num_of_robots)}

    glas_instance = CollisionAvoidanceSystem()

    mapped_conditions = process_initial_conditions(state)
    # print("Mapped conditions: ")
    # print(mapped_conditions)
    keys_list_times = list(mapped_conditions.keys())

    local_dist_victim = float('inf')
    local_pos_list_for_dist_victim = []
    local_dist_any = float('inf')
    local_pos_list_for_dist_any = []
    local_min_obstacle_dist_of_victim = float('inf')
    local_pos_list_for_obstacle_dist_of_victim = []
    local_min_obstacle_dist_of_any_robot = float('inf')
    local_pos_list_for_obstacle_dist_of_any_robot = []
    local_min_inter_robot_dist_of_victim_robot = float('inf')
    local_pos_list_for_inter_robot_dist_of_victim_robot = []
    local_min_inter_robot_dist_of_any_robot = float('inf')
    local_pos_list_for_inter_robot_dist_of_any_robot = []
    local_min_deadlock_of_victim_robot = float('inf')
    local_pos_list_for_deadlock_of_victim_robot = []
    local_min_deadlock_of_any_robot = float('inf')
    local_pos_list_for_deadlock_of_any_robot = []
    local_min_navigation_duration_of_victim_robot = float('inf')
    local_pos_list_for_navigation_duration_of_victim_robot = []
    local_max_navigation_duration_of_any_robot = -float('inf')
    local_pos_list_for_navigation_duration_of_any_robot = []

    delay_constant = 1
    # check if the attack is navigation delay attack
    if attackType == 9 or attackType == 10:
        delay_constant = delayConstant

    try:
        temp_list_for_global = []
        temp_list_for_global_prev = []
        temp_list_for_global_fake = []
        temp_list_for_global_with_velocities = []

        # run glas model with timestep
        ii = -1
        global totalTimeStep
        totalTimeStep = glas_instance.param.sim_times[-1]
        print("values: ", glas_instance.param.sim_times[-1],  delay_constant)
        while counter < (glas_instance.param.sim_times[-1] * delay_constant) - 1:
            ii += 1
            # print("girdim")

            # print("=======")
            # print("Counter: ", counter)

            all_pos = np.copy(glas_instance.get_agent_positions())
            # print("All positions: ", all_pos)
            updated_positions = []
            for pos in all_pos:
                updated_positions.append((pos[0], pos[1]))
            # print("Updated Positions: ", updated_positions)

            temp_pos = np.copy(
                glas_instance.get_agent_positions()[attackedRobotId])
            prev_pos = (temp_pos[0], temp_pos[1])
            # print("girdim")
            # print("Prev pos: ", prev_pos)

            # print real logs
            # temp_list_for_global_prev.append(counter)
            # for updated_position in updated_positions:
            #     temp_list_for_global_prev.append(updated_position)

            # plan attack
            if counter in keys_list_times:
                print("Attack!!")
                value = mapped_conditions[counter]
                # print("Value: ", value)
                # print("Updated positions: ", updated_positions)
                # print("Attacked robot id: ", attackedRobotId)
                updated_positions[attackedRobotId] = (
                    updated_positions[attackedRobotId][0] + value[0], updated_positions[attackedRobotId][1] + value[1])
                # print("Updated positions: ", updated_positions)
                glas_instance.set_agent_position(
                    attackedRobotId, updated_positions[attackedRobotId][0], updated_positions[attackedRobotId][1])
            # print("girdim")
            # print fake logs
            # temp_list_for_global_fake.append(counter)
            # for updated_position in updated_positions:
            #     temp_list_for_global_fake.append(updated_position)

            # glas_instance.calculate_next_velocities_(attackedRobotId, prev_pos)
            result = glas_instance.calculate_next_velocities_(
                attackedRobotId, prev_pos)
            SimResult = namedtuple(
                'SimResult', ['states', 'observations', 'actions', 'steps'])
            sim_result = SimResult._make(result)
            glas_instance.sim_results.append(sim_result)

            all_pos = np.copy(glas_instance.get_agent_positions())
            # print("All positions: ", all_pos)
            updated_positions = []
            for pos in all_pos:
                updated_positions.append((pos[0], pos[1]))
            # print("girdim")
            # print("Updated Positions: ", updated_positions)

            # glas_instance.write_positions_to_file(counter)

            # ==========================================
            # Attack measurements
            # ==========================================

            # ==========================================
            # decide collision of victim robot with obstacles:

            # print("R2O Attack")
            min_distance_to_obstacle = float('inf')
            min_dist_to_obstacle = calculate_min_distance_to_obstacles(
                updated_positions[victimRobotId],
                obstacles
            )
            min_distance_to_obstacle = min(
                min_distance_to_obstacle, min_dist_to_obstacle)

            # save for falsification
            list_min_obstacle_dist_of_victim.append(min_distance_to_obstacle)

            local_min_obstacle_dist_of_victim = min(
                local_min_obstacle_dist_of_victim, min_distance_to_obstacle)
            local_pos_list_for_obstacle_dist_of_victim.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_obstacle_dist_of_victim.append(
                    updated_position)

            # ==========================================
            # decide collision of victim robot with other robots:

            # print("R2R Attack")
            min_dist_of_inter_robot_to_victim_robot = float('inf')
            victim_position = updated_positions[victimRobotId]
            for robot_id, robot_pos in enumerate(updated_positions):
                if robot_id == victimRobotId or robot_id == attackedRobotId:
                    continue  # Skip the victim robot

                # Calculate the Euclidean distance
                min_inter_robot_dist = ((robot_pos[0] - victim_position[0]) ** 2 +
                                        (robot_pos[1] - victim_position[1]) ** 2) ** 0.5
                # check all robots
                min_dist_of_inter_robot_to_victim_robot = min(
                    min_dist_of_inter_robot_to_victim_robot, min_inter_robot_dist)

                # print for me
                # if min_dist_of_inter_robot_to_victim_robot < radius * 2:
                # print("Collision with other robots at time:", counter, " between robots",
                #   victimRobotId, "and", robot_id, "with positions", victim_position, robot_pos)
            # save for falsification
            list_min_inter_robot_dist_of_victim_robot.append(
                min_dist_of_inter_robot_to_victim_robot)

            local_min_inter_robot_dist_of_victim_robot = min(
                local_min_inter_robot_dist_of_victim_robot, min_dist_of_inter_robot_to_victim_robot)
            local_pos_list_for_inter_robot_dist_of_victim_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_inter_robot_dist_of_victim_robot.append(
                    updated_position)

            # ==========================================
            # decide herding attack for victim robot
            # dist: difference of static point and victim's position
            # print("Herding Attack")
            dist_victim_point = calculate_abs_distance(pointX, pointY,
                                                       updated_positions[victimRobotId][0], updated_positions[victimRobotId][1])
            # save for falsification
            list_min_navigating_to_point_victim.append(dist_victim_point)

            # save to print
            local_dist_victim = min(local_dist_victim, dist_victim_point)
            local_pos_list_for_dist_victim.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_dist_victim.append(updated_position)

            # ==========================================
            # decide deadlock of victim robot:
            # print("Deadlock Attack")
            last_positions.append(updated_positions[victimRobotId])
            # Keep only the last positions
            if len(last_positions) > deadlockTimestep:
                last_positions.pop(0)

            total_pos_change = 0.0
            if len(last_positions) == deadlockTimestep:
                # sum of the move changes between consecutive positions in last_positions
                for i in range(len(last_positions) - 1):
                    total_pos_change += calculate_abs_distance(
                        last_positions[i][0], last_positions[i][1], last_positions[i+1][0], last_positions[i+1][1])
            else:
                total_pos_change = float('inf')

            # edge case
            # skip last moments if the robot is close to the goal within 10 meters and does not move more.
            if calculate_abs_distance(updated_positions[victimRobotId][0], updated_positions[victimRobotId][1], goal_positions[victimRobotId][0], goal_positions[victimRobotId][1]) < 5.0:
                total_pos_change = float('inf')

            # print for me
            # if total_pos_change < deadlockPosChange:
                # print("Deadlock of victim robot at time:", counter, " with position",
                #   updated_positions[victimRobotId], " and measurement: ", total_pos_change)

            # save for falsification
            list_min_deadlock_of_victim_robot.append(
                total_pos_change)

            # save to print
            local_min_deadlock_of_victim_robot = min(
                local_min_deadlock_of_victim_robot, total_pos_change)
            local_pos_list_for_deadlock_of_victim_robot.append(counter)
            for updated_position in updated_positions:
                local_pos_list_for_deadlock_of_victim_robot.append(
                    updated_position)

            # ==========================================
            # decide navigation delay of victim robot:

            # print("Navigation Delay Attack")
            current_position = updated_positions[victimRobotId]
            remaining_distance = calculate_abs_distance(
                current_position[0], current_position[1], goal_positions[victimRobotId][0], goal_positions[victimRobotId][1])

            # save for falsification
            list_min_navigation_duration_of_victim_robot.append(
                remaining_distance)

            # save to print
            local_min_navigation_duration_of_victim_robot = min(
                local_min_navigation_duration_of_victim_robot, remaining_distance)
            local_pos_list_for_navigation_duration_of_victim_robot.append(
                counter)
            for updated_position in updated_positions:
                local_pos_list_for_navigation_duration_of_victim_robot.append(
                    updated_position)

            # ==========================================
            temp_list_for_global.append(counter)
            temp_list_for_global_with_velocities.append(counter)
            for updated_position in updated_positions:
                temp_list_for_global.append(updated_position)
                temp_list_for_global_with_velocities.append(updated_position)
            # for updated_velocity in updated_velocities:
                # temp_list_for_global_with_velocities.append(updated_velocity)

            time_result.append(counter)
            # counter += time_step
            counter = increment_precise(
                counter, glas_instance.param.sim_dt, 10)
            # print("Counter: ", counter)

        if attackType == 1:
            # victim - static obstacle collision
            if local_min_obstacle_dist_of_victim < global_min_dist_of_victim_to_static_so_far:
                global_min_dist_of_victim_to_static_so_far = local_min_obstacle_dist_of_victim
                global_log_list = local_pos_list_for_obstacle_dist_of_victim
        elif attackType == 3:
            # victim robot - other robots collision
            if local_min_inter_robot_dist_of_victim_robot < global_min_dist_of_victim_to_robots_so_far:
                global_min_dist_of_victim_to_robots_so_far = local_min_inter_robot_dist_of_victim_robot
                global_log_list = local_pos_list_for_inter_robot_dist_of_victim_robot
        elif attackType == 5:
            # navigating the point - victim robot
            if local_dist_victim < global_min_navigating_to_point_victim:
                global_min_navigating_to_point_victim = local_dist_victim
                global_log_list = local_pos_list_for_dist_victim
        elif attackType == 7:
            # victim robot - deadlock
            if local_min_deadlock_of_victim_robot < global_min_deadlock_of_victim_robot:
                global_min_deadlock_of_victim_robot = local_min_deadlock_of_victim_robot
                global_log_list = local_pos_list_for_deadlock_of_victim_robot
        elif attackType == 9:
            # victim robot - navigation delay
            if local_min_navigation_duration_of_victim_robot < global_min_navigation_duration_of_victim_robot:
                global_min_navigation_duration_of_victim_robot = local_min_navigation_duration_of_victim_robot
                global_log_list = local_pos_list_for_navigation_duration_of_victim_robot

    except Exception as e:
        print("Errors related to falsification!")
        print(e)
    try:
        print("Counter: ", counter)
        print("global_iteration: ", global_iteration)
        print("time_result:", time_result, len(time_result))
        print("list_min_obstacle_dist_of_victim:",
              list_min_obstacle_dist_of_victim, len(
                  list_min_obstacle_dist_of_victim)
              )
        print("list_min_obstacle_dist_of_any_robot:",
              list_min_obstacle_dist_of_any_robot, len(list_min_obstacle_dist_of_any_robot))
        print("list_min_inter_robot_dist_of_victim_robot:",
              list_min_inter_robot_dist_of_victim_robot, len(list_min_inter_robot_dist_of_victim_robot))
        print("list_min_inter_robot_dist_of_any_robot:",
              list_min_inter_robot_dist_of_any_robot, len(
                  list_min_inter_robot_dist_of_any_robot))
        print("list_min_navigating_to_point_victim:",
              list_min_navigating_to_point_victim, len(list_min_navigating_to_point_victim))
        print("list_min_navigating_to_point_any:",
              list_min_navigating_to_point_any, len(list_min_navigating_to_point_any))
        print("list_min_deadlock_of_victim_robot:",
              list_min_deadlock_of_victim_robot, len(list_min_deadlock_of_victim_robot))
        print("list_min_deadlock_of_any_robot:",
              list_min_deadlock_of_any_robot, len(list_min_deadlock_of_any_robot))
        print("list_min_navigation_duration_of_victim_robot:",
              list_min_navigation_duration_of_victim_robot, len(list_min_navigation_duration_of_victim_robot))
        print("list_max_navigation_duration_of_any_robot:",
              list_max_navigation_duration_of_any_robot, len(list_max_navigation_duration_of_any_robot))
        print("list_min_deadlock_for_navigation_duration_of_any_robot:",
              list_min_deadlock_for_navigation_duration_of_any_robot, len(list_min_deadlock_for_navigation_duration_of_any_robot))

        trace = Trace(time_result, [time_result,
                                    list_min_obstacle_dist_of_victim,
                                    [0.0] * len(time_result),
                                    list_min_inter_robot_dist_of_victim_robot,
                                    [0.0] * len(time_result),
                                    list_min_navigating_to_point_victim,
                                    [0.0] * len(time_result),
                                    list_min_deadlock_of_victim_robot,
                                    [0.0] * len(time_result),
                                    list_min_navigation_duration_of_victim_robot,
                                    [0.0] * len(time_result),
                                    [0.0] * len(time_result)])
        # trace = Trace(time_result, [time_result,
        #                             list_min_obstacle_dist_of_victim,
        #                             [0.0] * len(time_result),
        #                             list_min_inter_robot_dist_of_victim_robot,
        #                             [0.0] * len(time_result),
        #                             list_min_navigating_to_point_victim,
        #                             [0.0] * len(time_result),
        #                             list_min_deadlock_of_victim_robot,
        #                             [0.0] * len(time_result),
        #                             list_min_navigation_duration_of_victim_robot,
        #                             [0.0] * len(time_result),
        #                             [0.0] * len(time_result)])
        global_log_list_last = temp_list_for_global
        global_log_list_prev = temp_list_for_global_prev
        global_log_list_fake = temp_list_for_global_fake
        global_log_list_last_with_velocities = temp_list_for_global_with_velocities
        # print("global_log_list_last: ", global_log_list_last)
        # print("result:")
        # print(trace)
        return BasicResult(trace)
    except Exception as e:
        print("Exception occurred during simulation")
        return FailureResult()


state_matcher = {
    "t": 0,
    "min_dist_of_victim_to_obstacles": 1,
    "min_dist_of_any_robot_to_obstacles": 2,
    "min_of_inter_robot_dis_of_victim_robot": 3,
    "min_of_inter_robot_dis_of_any_robot": 4,
    "min_dist_victim": 5,
    "min_dist_any": 6,
    "deltaPos_VictimRobot": 7,
    "deltaPos_AnyRobot": 8,
    "distToGoal_VictimRobot": 9,
    "distToGoal_AnyRobot": 10,
    "deltaPos_Any_for_NavigationDelay": 11
}

# Decide attack type
specification = None
attackName = ""
if attackType == 1:
    attackName = "collision_obstacle_victim_robot"
    phi_collision_obstacle_victim_drone = f"always (min_dist_of_victim_to_obstacles > {radius})"
    print("phi_collision_obstacle_targeted_drone: ",
          phi_collision_obstacle_victim_drone)
    specification = RTAMTDense(
        phi_collision_obstacle_victim_drone, state_matcher)
elif attackType == 3:
    attackName = "collision_btw_victim_and_other_robots"
    phi_collision_btw_victim_and_other_robots = f"always (min_of_inter_robot_dis_of_victim_robot > {radius*2})"
    print("phi_collision_btw_victim_and_other_robots: ",
          phi_collision_btw_victim_and_other_robots)
    specification = RTAMTDense(
        phi_collision_btw_victim_and_other_robots, state_matcher)
elif attackType == 5:
    attackName = "navigating_to_point_targeted"
    phi_navigating_robot_targeted = "always (min_dist_victim > 0.5)"
    print("phi_navigating_victim: ", phi_navigating_robot_targeted)
    specification = RTAMTDense(
        phi_navigating_robot_targeted, state_matcher)
elif attackType == 7:
    attackName = "deadlock_of_victim_robot"
    phi_deadlock_of_victim_robot = f"always deltaPos_VictimRobot > {deadlockPosChange}"
    print("phi_deadlock_of_victim_robot: ", phi_deadlock_of_victim_robot)
    specification = RTAMTDense(
        phi_deadlock_of_victim_robot, state_matcher)
elif attackType == 9:
    attackName = "navigation_delay_of_victim_robot"
    phi_navigation_delay_of_victim_robot = f"eventually (distToGoal_VictimRobot <= 1.0)"
    # phi_navigation_delay_of_victim_robot = f"eventually (distToGoal_VictimRobot <= 1.0)"
    print("phi_navigation_delay_of_victim_robot: ",
          phi_navigation_delay_of_victim_robot)
    specification = RTAMTDense(
        phi_navigation_delay_of_victim_robot, state_matcher)
else:
    print("Invalid attack type!")
    exit()

# Construct source and destination paths
# TODO: change the path to the config file
base_path = "../comparison/swarmlab/glas_ws/results/singleintegrator/instances"
print("yaml_name: ", yaml_name)
source_yaml = os.path.join(base_path, f"{yaml_name}.yaml")
dest_yaml = os.path.join(base_path, "map_8by8_obst12_agents8_ex0000.yaml")

# Copy the file
print(f"Attempting to copy from: {source_yaml}")
shutil.copy2(source_yaml, dest_yaml)
print(f"Copied {source_yaml} to {dest_yaml}")

print_all_config_params()
print("Total time step: ", totalTimeStep)

# message spoofing variables
initial_conditions = [
    (0, totalTimeStep),
    (-2, 2),
    (-2, 2)
] * numberOfFalseMessage

optimizer = DualAnnealing()
flag = 1
i = -1
while i < falsificationRuns and flag:
    i += 1
    options = Options(runs=1, iterations=falsificationIterations, interval=(
        0, 10), static_parameters=initial_conditions, seed=random.randint(0, 2**32 - 1))
    result = staliro(Glas_Model, specification, optimizer, options)

    worst_run_ = worst_run(result)
    worst_sample = worst_eval(worst_run_).sample
    worst_result = simulate_model(Glas_Model, options, worst_sample)

    print("\nWorst Sample:")
    print(worst_sample)

    print("\nResult:")
    print(worst_result.trace.states)

    print("\nWorst Evaluation:")
    print(worst_eval(worst_run(result)))

    print("Run: ", i)

    if worst_eval(worst_run_).cost < 0:
        flag = 0

        # with open(f'/Users/doguhanyeke/Desktop/research/False_Data_Attacks_on_ORCA/Attack/src/falsiResults2/{attackName}_run_positions_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}_{algorithm}_{radius}_{time_step}.txt', 'w') as file:
        #     for i in range(0, len(global_log_list), num_of_robots + 1):
        #         time = global_log_list[i]
        #         positions = global_log_list[i+1:i+num_of_robots+1]
        #         positions_str = ' '.join(
        #             [f"({pos[0]:.4f},{pos[1]:.4f})" for pos in positions])
        #         file.write(f"{time} {positions_str}\n")
        # print("End")

        # if evaluation cost is negative, then print
    if worst_eval(worst_run_).cost < 0:
        print("Found the attack!")
        print("Evaluation cost: ", worst_eval(worst_run_).cost)
        with open(f'../False_Data_Attacks_on_ORCA/Attack/src/falsiParams/{attackName}_ite_{numberOfFalseMessage}mes_{environment_name}_{algorithm}_{radius}_{time_step}.txt', 'w') as file:
            file.write(str(worst_eval(worst_run(result))))

        with open(f'../False_Data_Attacks_on_ORCA/Attack/src/falsiResults3/{attackName}_run_positions_{falsificationIterations}ite_{numberOfFalseMessage}mes_{environment_name}_{algorithm}_{radius}_{time_step}.txt', 'w') as file:
            for i in range(0, len(global_log_list_last), num_of_robots + 1):
                time = global_log_list_last[i]
                positions = global_log_list_last[i+1:i+num_of_robots+1]
                positions_str = ' '.join(
                    [f"({pos[0]:.4f},{pos[1]:.4f})" for pos in positions])
                file.write(f"{time} {positions_str}\n")
        print("End2")
    else:
        print("Could not find an attack!")


def point_line_distance(point, line_start, line_end):
    """Calculate distance from a point to a line segment."""
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    line_vec = line_end - line_start
    point_vec = point - line_start
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        return np.linalg.norm(point_vec)

    t = max(0, min(1, np.dot(point_vec, line_vec) / (line_length * line_length)))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def get_square_corners(bottom_left):
    """Convert bottom-left corner point into four corners of 1x1 square."""
    x, y = bottom_left
    return [
        (x, y),        # bottom-left
        (x + 1, y),    # bottom-right
        (x + 1, y + 1),  # top-right
        (x, y + 1)     # top-left
    ]


def calculate_min_distance_to_obstacles(point, obstacles):
    """Calculate minimum distance from a point to any obstacle."""
    min_dist = float('inf')

    for obstacle in obstacles:
        corners = get_square_corners(obstacle)
        for i in range(len(corners)):
            p1 = corners[i]
            p2 = corners[(i + 1) % len(corners)]
            dist = point_line_distance(point, p1, p2)
            min_dist = min(min_dist, dist)

    return min_dist
