import numpy as np
import orca_module
from typing import List, Tuple
import os
import matlab.engine
import csv


class OrcaAttackWrapper:
    def __init__(self, num_robots: int, time_step: float, obstacle_pos: List[Tuple[float, float]], seed: int):
        self.num_robots = num_robots
        self.time_step = time_step
        self.obstacle_pos = obstacle_pos
        self.obstacle_radius = 0.5
        self.agent_radius = 0.25
        self.seed = seed

        # Initialize ORCA
        self.orca_instance = orca_module.init_orca()

        # Initialize MATLAB engine
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()

        # Setup paths first
        self.root_dir = os.path.join(os.getcwd(), '../')
        self.fuzz_dir = os.path.join(self.root_dir, 'fuzz')

        # Add MATLAB paths
        self.eng.addpath(self.fuzz_dir)
        self.eng.addpath(os.path.join(self.fuzz_dir, 'prepare'))
        self.eng.addpath(os.path.join(self.fuzz_dir, 'search'))
        self.eng.addpath(os.path.join(self.fuzz_dir, 'seedpools'))

        # Initialize file paths with seed
        self.setup_data_files(seed)

    def __del__(self):
        """Cleanup MATLAB engine when done"""
        if hasattr(self, 'eng'):
            self.eng.quit()

    def setup_data_files(self, seed: int):
        """Setup CSV files for data collection with seed-specific names"""
        # Use seed in filenames
        self.pos_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/attack_pos{seed}.csv')
        self.dist_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/dist_obs{seed}.csv')
        self.col_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/nb_col{seed}.csv')
        self.dire_csv = os.path.join(
            self.fuzz_dir, f'tmp_files/dire{seed}.csv')

        # Create directories if they don't exist
        os.makedirs(os.path.join(self.fuzz_dir, 'tmp_files'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'search'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'prepare'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'seedpools'), exist_ok=True)
        os.makedirs(os.path.join(self.fuzz_dir, 'attResults'), exist_ok=True)

    def record_state(self, time: float):
        """Record current state for attack analysis"""
        positions = orca_module.get_positions(self.orca_instance)
        velocities = orca_module.get_velocities(self.orca_instance)

        # Record positions
        with open(self.pos_csv, 'a') as f:
            pos_line = f"{time}"
            for pos in positions:
                pos_line += f" {pos[0]:.4f} {pos[1]:.4f}"
            f.write(pos_line + "\n")

        # Record distances to obstacles
        with open(self.dist_csv, 'a') as f:
            for i, pos in enumerate(positions):
                distances = self.calculate_agent_obstacle_distances(pos)
                min_dist = min(distances)
                f.write(f"{time} {i} {min_dist:.4f}\n")

        # Record collisions
        collisions = self.detect_collisions(positions)
        with open(self.col_csv, 'a') as f:
            f.write(f"{time} {int(collisions)}\n")

        # Record directions (relative to obstacles)
        with open(self.dire_csv, 'a') as f:
            for i, pos in enumerate(positions):
                direction = self.calculate_direction(pos, velocities[i])
                f.write(f"{time} {i} {direction}\n")

    def calculate_agent_obstacle_distances(self, pos) -> List[float]:
        """Calculate distances to all obstacles for an agent"""
        distances = []
        for obs_pos in self.obstacle_pos:
            dist = np.sqrt((pos[0] - obs_pos[0])**2 +
                           (pos[1] - obs_pos[1])**2) - \
                (self.obstacle_radius + self.agent_radius)
            distances.append(dist)
        return distances

    def detect_collisions(self, positions) -> bool:
        """Detect any collisions with obstacles"""
        for pos in positions:
            for obs_pos in self.obstacle_pos:
                dist = np.sqrt((pos[0] - obs_pos[0])**2 +
                               (pos[1] - obs_pos[1])**2)
                if dist < (self.obstacle_radius + self.agent_radius):  # Using both radii
                    return True
        return False

    def calculate_direction(self, pos, vel) -> int:
        """
        Calculate if agent is passing obstacle from left (-1) or right (1)
        Based on the agent's position, velocity, and closest obstacle
        """
        # Find closest obstacle
        closest_obs = None
        min_dist = float('inf')
        for obs_pos in self.obstacle_pos:
            dist = np.sqrt((pos[0] - obs_pos[0])**2 + (pos[1] - obs_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs_pos

        if closest_obs is None:
            return 0  # No obstacle case

        # Calculate cross product to determine if passing left or right
        # Vector from obstacle to agent
        to_agent = [pos[0] - closest_obs[0], pos[1] - closest_obs[1]]

        # Use cross product with velocity to determine side
        cross_product = to_agent[0] * vel[1] - to_agent[1] * vel[0]

        return 1 if cross_product > 0 else -1  # 1 for right, -1 for left

    def spoof_position(self, agent_id: int, dev_y: float,
                       start_t: float, duration: float, current_time: float):
        """Apply GPS spoofing using ORCA's native position handling"""
        positions = orca_module.get_positions(self.orca_instance)
        velocities = orca_module.get_velocities(self.orca_instance)

        if start_t <= current_time < (start_t + duration):
            # Create spoofed position
            spoofed_pos = positions[agent_id].copy()
            spoofed_pos[1] += dev_y

            # Use ORCA's native spoofing mechanism
            orca_module.calculate_next_positions(
                self.time_step,
                positions,
                velocities,
                agent_id,  # spoofed robot id
                spoofed_pos,  # spoofed position
                self.orca_instance
            )
        else:
            # Normal update
            orca_module.calculate_next_positions(
                self.time_step,
                positions,
                velocities,
                -1,  # no spoofing
                positions[0],  # dummy position (not used)
                self.orca_instance
            )

    def record_distances(self, current_time: float):
        """Record minimum distance to obstacles for each robot"""
        distances = []
        for i in range(self.num_robots):
            pos = orca_module.get_position(self.orca_instance, i)
            # Find minimum distance to any obstacle
            min_dist = float('inf')
            for obs_pos in self.obstacle_pos:
                dist = np.sqrt((pos[0] - obs_pos[0])**2 +
                               (pos[1] - obs_pos[1])**2) - self.obstacle_radius
                min_dist = min(min_dist, dist)
            distances.append([current_time, i, min_dist])

        with open(self.dist_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(distances)

    def run_benign_simulation(self, simulation_time: float):
        """Run simulation without attack to collect initial data"""
        print(f"Running benign simulation for {simulation_time} seconds...")
        current_time = 0.0
        while current_time < simulation_time:
            self.record_state(current_time)
            # Update ORCA without spoofing
            orca_module.step(self.orca_instance)
            current_time += self.time_step

    def run_attack_simulation(self, target_robot: int, spoofed_robot: int,
                              dev: float, start_t: float, duration: float):
        """Run simulation with GPS spoofing attack"""
        print(f"Running attack simulation: spoofing robot {spoofed_robot} "
              f"to attack robot {target_robot}")

        # First run benign simulation
        self.run_benign_simulation(30.0)  # 30 seconds of benign data

        # Then run attack analysis
        self.run_attack_analysis(dev)

    def run_attack_analysis(self, dev: float):
        """Run attack analysis using MATLAB functions"""
        try:
            print("Running prepare phase...")
            [start_t, pos_att, dist_obs, final_dire] = self.eng.prepare(
                float(self.num_robots),
                matlab.double(self.obstacle_pos),
                self.pos_csv,
                self.dist_csv,
                self.dire_csv,
                self.col_csv,
                float(self.seed),
                nargout=4
            )

            if start_t > -1:
                print("Running seed generation...")
                seedpool_csv = os.path.join(self.fuzz_dir,
                                            f'seedpools/pool{self.seed}.csv')
                success = self.run_seed_generation(start_t, pos_att, dist_obs,
                                                   final_dire, seedpool_csv)

                if success:
                    print("Running gradient descent...")
                    self.run_gradient_descent(seedpool_csv, dev)
            else:
                print("No suitable attack point found in prepare phase")

        except Exception as e:
            print(f"Error in attack analysis: {str(e)}")
            raise

    def run_seed_generation(self, start_t, pos_att, dist_obs, final_dire,
                            seedpool_csv):
        """Run seed generation phase"""
        try:
            flag = self.eng.seed_gen(
                float(self.num_robots),
                float(start_t),
                matlab.double(pos_att),
                matlab.double(dist_obs),
                matlab.double(final_dire),
                seedpool_csv,
                nargout=1
            )
            return os.path.exists(seedpool_csv)
        except Exception as e:
            print(f"Error in seed generation: {str(e)}")
            return False

    def run_gradient_descent(self, seedpool_csv: str, dev: float):
        """Run gradient descent phase"""
        try:
            f_out = os.path.join(self.fuzz_dir,
                                 f'attResults/att_results{self.seed}.csv')
            self.eng.search_grad(
                seedpool_csv,
                float(dev),
                20.0,  # max_iterations
                f_out,
                0.1,   # duration
                float(self.seed),
                self.pos_csv,
                self.dist_csv,
                self.col_csv,
                os.path.join(
                    self.fuzz_dir, f'tmp_files/col_info{self.seed}.csv'),
                os.path.join(
                    self.fuzz_dir, f'tmp_files/parent{self.seed}.csv'),
                os.path.join(
                    self.fuzz_dir, f'tmp_files/neighbor{self.seed}.csv'),
                os.path.join(
                    self.fuzz_dir, f'search/iteration{self.seed}.csv'),
                os.path.join(self.fuzz_dir, f'search/condition{self.seed}.csv')
            )
        except Exception as e:
            print(f"Error in gradient descent: {str(e)}")
            raise


def main():
    """
    Run attack simulation with parameters matching the original SwarmFuzz setup
    Args similar to swarmfuzz(200, 201, 5, 5):
    - seedStart, seedEnd: 200, 201 for mission seeds
    - dev: 5 (GPS spoofing deviation in meters)
    - nb: 5 (number of drones in swarm)
    """
    # Attack parameters matching their setup
    seedStart = 200
    seedEnd = 201
    dev = 5.0     # 5m GPS spoofing deviation
    num_robots = 3  # 5 drones in swarm
    time_step = 0.5
    max_iterations = 20
    duration = 0.1  # Initial attack duration

    # Obstacle positions (centers)
    obstacle_pos = [
        (25.5, 15.5),
        (26.5, 15.5),
        (21.5, 17.5),
        (20.5, 17.5),
        (29.5, 13.5),
        (30.5, 13.5),
        (31.5, 13.5)
    ]

    # Setup file paths
    root_dir = os.path.join(os.getcwd(), '../')

    for seed in range(seedStart, seedEnd + 1):
        # Temp files
        pos_csv = os.path.join(
            root_dir, f'fuzz/tmp_files/attack_pos{seed}.csv')
        dist_csv = os.path.join(root_dir, f'fuzz/tmp_files/dist_obs{seed}.csv')
        col_csv = os.path.join(root_dir, f'fuzz/tmp_files/nb_col{seed}.csv')
        info_csv = os.path.join(root_dir, f'fuzz/tmp_files/col_info{seed}.csv')
        parent_csv = os.path.join(root_dir, f'fuzz/tmp_files/parent{seed}.csv')
        neigh_csv = os.path.join(
            root_dir, f'fuzz/tmp_files/neighbor{seed}.csv')

        # Output files
        ite_csv = os.path.join(root_dir, f'fuzz/search/iteration{seed}.csv')
        cond_csv = os.path.join(root_dir, f'fuzz/search/condition{seed}.csv')
        dire_csv = os.path.join(root_dir, f'fuzz/prepare/dire{seed}.csv')
        seedpool_csv = os.path.join(root_dir, f'fuzz/seedpools/pool{seed}.csv')
        f_out = os.path.join(
            root_dir, f'fuzz/attResults/att_results{seed}.csv')

        # Initialize wrapper
        wrapper = OrcaAttackWrapper(
            num_robots=num_robots,
            time_step=time_step,
            obstacle_pos=obstacle_pos,
            seed=seed  # Added the seed parameter
        )

        print(f"********** No attack. Running seed: {seed} **********")

        # Phase 1: Run without attack to collect initial data
        simulation_time = 30.0
        current_time = 0.0

        while current_time < simulation_time:
            wrapper.record_state(current_time)
            wrapper.spoof_position(
                agent_id=-1,  # No spoofing
                dev_y=0,
                start_t=0,
                duration=0,
                current_time=current_time
            )
            current_time += time_step

        # Phase 2: Call MATLAB prepare function
        [start_t, pos_att, dist_obs, final_dire] = wrapper.eng.prepare(
            float(num_robots),
            matlab.double(obstacle_pos),
            pos_csv,
            dist_csv,
            dire_csv,
            col_csv,
            float(seed),
            nargout=4
        )

        if start_t > -1:  # If no collisions in benign run
            print("*********** Swarm vulnerability graph generator **********")
            # Call MATLAB seed generation
            flag = wrapper.eng.seed_gen(
                float(num_robots),
                float(start_t),
                matlab.double(pos_att),
                matlab.double(dist_obs),
                matlab.double(final_dire),
                seedpool_csv,
                nargout=1
            )

            if os.path.exists(seedpool_csv):
                print("*********** Gradient descent ***********")
                # Call MATLAB gradient descent
                wrapper.eng.search_grad(
                    seedpool_csv,
                    float(dev),
                    float(max_iterations),
                    f_out,
                    float(duration),
                    float(seed),
                    pos_csv,
                    dist_csv,
                    col_csv,
                    info_csv,
                    parent_csv,
                    neigh_csv,
                    ite_csv,
                    cond_csv
                )


if __name__ == "__main__":
    main()
