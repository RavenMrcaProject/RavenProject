import os
import numpy as np
from typing import List, Tuple
from collections import namedtuple
import csv
from examples.run_singleintegrator import SingleIntegratorParam
from benign_glas_sample import CollisionAvoidanceSystem
import torch
import yaml


class GlasAttackWrapper:
    def __init__(self, num_robots: int, time_step: float, seed: int):
        # Load obstacle positions from YAML file
        # TODO: change the path to the config file
        base_dir = '../comparison/swarmlab/glas_ws'
        yaml_path = os.path.join(base_dir,
                                 'results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml')

        with open(yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)
            self.obstacle_pos = map_data['map']['obstacles']

        self.num_robots = num_robots
        self.time_step = time_step
        self.seed = seed

        # Initialize GLAS using existing implementation
        print("Initializing GLAS...")
        self.glas = CollisionAvoidanceSystem()

        # Setup paths for attack data
        self.root_dir = os.path.join(os.getcwd(), '../')
        self.fuzz_dir = os.path.join(self.root_dir, 'fuzz')

        # Initialize data files
        self.setup_data_files(seed)

    def setup_data_files(self, seed: int):
        """Setup CSV files for data collection"""
        os.makedirs(os.path.join(self.fuzz_dir, 'tmp_files'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'search'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'prepare'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'seedpools'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'attResults'), exist_ok=True)

        self.pos_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/attack_pos{seed}.csv')
        self.dist_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/dist_obs{seed}.csv')
        self.col_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/nb_col{seed}.csv')
        self.dire_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/dire{seed}.csv')

    def record_state(self, time: float):
        """Record current state for attack analysis"""
        positions = self.glas.get_agent_positions()

        # Record positions in the specified format
        with open(self.pos_csv, 'a') as f:
            pos_line = f"{time:.1f}"
            for pos in positions:
                # Adding a fixed z-coordinate of 5.0 to match the format
                pos_line += f" ({pos[0]:.4f},{pos[1]:.4f},5.0000)"
            f.write(pos_line + "\n")

        # Record distances to obstacles
        with open(self.dist_csv, 'a') as f:
            for i, pos in enumerate(positions):
                distances = self.calculate_agent_obstacle_distances(pos)
                min_dist = min(distances) if distances else float('inf')
                f.write(f"{time} {i} {min_dist:.4f}\n")

        # Record collisions
        collisions = self.detect_collisions()
        with open(self.col_csv, 'a') as f:
            f.write(f"{time} {int(collisions)}\n")

    def calculate_agent_obstacle_distances(self, pos) -> List[float]:
        """Calculate distances from agent to all obstacles"""
        distances = []
        for obs in self.glas.env.obstacles:
            dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
            distances.append(dist)
        return distances

    def detect_collisions(self) -> bool:
        """Check for collisions between agents and obstacles"""
        positions = self.glas.get_agent_positions()
        for pos in positions:
            for obs in self.glas.env.obstacles:
                dist = np.sqrt((pos[0] - obs[0])**2 + (pos[1] - obs[1])**2)
                if dist < (self.glas.param.r_agent + 0.5):  # Using GLAS's agent radius
                    return True
        return False

    def run_attack(self, attack_agent_id: int, attack_magnitude: float,
                   attack_start: float, attack_duration: float):
        """Run simulation with attack"""
        self.glas.delete_file()  # Clear previous position records

        current_time = 0
        step_counter = 0

        while not self.glas.reached_goal() and current_time < self.glas.param.sim_times[-1]:
            # Get current positions
            agent_pos = np.copy(
                self.glas.get_agent_positions()[attack_agent_id])
            agent_original_pos = np.copy(agent_pos)

            # Apply attack if in attack window
            if attack_start <= current_time <= (attack_start + attack_duration):
                spoofed_pos = np.copy(agent_pos)
                spoofed_pos[1] += attack_magnitude  # Modify Y coordinate
                self.glas.set_agent_position(
                    attack_agent_id, spoofed_pos[0], spoofed_pos[1])
                print(
                    f"Attack active - Original pos: {agent_original_pos}, Spoofed pos: {spoofed_pos}")

            # Calculate next step
            result = self.glas.calculate_next_velocities_(
                attack_agent_id, agent_pos)

            # Record positions
            self.glas.write_positions_to_file(current_time)

            # Store results
            SimResult = namedtuple(
                'SimResult', ['states', 'observations', 'actions', 'steps'])
            sim_result = SimResult._make(result)
            self.glas.sim_results.append(sim_result)

            current_time += self.glas.param.sim_dt
            step_counter += 1

            print("####################################")

        print(f"Reached Goal: {self.glas.reached_goal()}")
        print(f"Time: {current_time}")
        print(f"Steps: {step_counter}")
        print(f"Max Time: {self.glas.param.sim_times[-1]}")

        return self.glas.reached_goal(), current_time


def main():
    """Run attack simulation with GLAS"""
    # Attack parameters
    seed = 200
    num_robots = 3
    time_step = 0.5

    print(f"********** Running seed: {seed} **********")

    wrapper = GlasAttackWrapper(
        num_robots=num_robots,
        time_step=time_step,
        seed=seed
    )

    # Run attack simulation
    attack_params = {
        'attack_agent_id': 1,
        'attack_magnitude': 5.0,
        'attack_start': 10.0,
        'attack_duration': 2.0
    }

    reached_goal, final_time = wrapper.run_attack(**attack_params)

    print(f"Reached Goal: {reached_goal}")
    print(f"Final Time: {final_time}")

    # Visualize results
    # wrapper.glas.draw()


if __name__ == "__main__":
    main()
