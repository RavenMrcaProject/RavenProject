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

# Custom unpickler to handle module paths


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


def main():
    c = CollisionAvoidanceSystem()
    c.delete_file()

    i = 0
    ii = 0
    while not c.reached_goal() and i < c.param.sim_times[-1]:
        agent_id = 1
        agent_pos = np.copy(c.get_agent_positions()[agent_id])
        agent_original_pos = np.copy(c.get_agent_positions()[agent_id])

        # if ii == 300:
        #     break
        # if ii % 3 == 0:
        # spoofing
        # agent_pos[0] = 22.5
        # agent_pos[1] = 12.0
        c.set_agent_position(agent_id, agent_pos[0], agent_pos[1])

        print("agentPos: ", agent_pos)
        # Corrected print statement
        print("agentOriginalPos: ", agent_original_pos)

        result = c.calculate_next_velocities_(
            agent_id, agent_pos)

        c.print_agent_positions()
        c.print_agent_velocities()
        c.write_positions_to_file(i)

        SimResult = namedtuple(
            'SimResult', ['states', 'observations', 'actions', 'steps'])
        sim_result = SimResult._make(result)
        c.sim_results.append(sim_result)

        print("####################################")
        i += c.param.sim_dt
        ii += 1

    print("Reached Goal:", c.reached_goal())
    print(i)
    print(ii)
    print(c.param.sim_times[-1])


main()
