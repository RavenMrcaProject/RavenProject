import matlab.engine
import os
import sys
import numpy as np
from configparser import ConfigParser
import orca_module
import csv


class OrcaSwarmFuzzWrapper:
    def __init__(self, target_obstacle_idx=0):
        # Base paths
        self.base_path = "../comparison/swarmlab"
        self.fuzz_dir = os.path.join(self.base_path, 'fuzz')
        self.orca_dir = os.path.join(self.base_path, 'orca')

        # Environment and config paths
        self.env_path = "../False_Data_Attacks_on_ORCA/Attack/environments/env-orca-1"
        self.config_path = "../False_Data_Attacks_on_ORCA/Attack/src/config.txt"

        # File paths
        self.obstacles_file = os.path.join(self.env_path, 'obstacles.txt')
        self.goals_file = os.path.join(self.env_path, 'goals.txt')
        self.init_pos_file = os.path.join(self.env_path, 'robot_pos.txt')

        # Target obstacle index
        self.target_obstacle_idx = target_obstacle_idx

        # Initialize MATLAB engine
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.fuzz_dir)

        # Parse config
        self.config = self.parse_config()

        # Initialize parameters
        self.nb_robots = int(self.config.get('num_robots', 3))
        self.time_step = float(self.config.get('time_step', 0.5))

        # SwarmFuzz parameters
        self.seed_start = 200
        self.seed_end = 201
        self.dev = 5

        # Output files
        self.benign_log = os.path.join(self.orca_dir, 'benign_results.txt')
        self.attack_log = os.path.join(self.orca_dir, 'attack_results.txt')

        # ORCA instance
        self.orca_instance = None
        self.attack_params = None

    def parse_config(self):
        config = {}
        try:
            with open(self.config_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(
                        ), parts[1].strip().strip('"')
                        config[key] = value
        except Exception as e:
            print(f"Error reading config file from {self.config_path}: {e}")
            raise
        return config

    def load_environment(self):
        """Load ORCA environment data"""
        try:
            # Load obstacles
            print(f"Loading obstacles from: {self.obstacles_file}")
            obstacles = []
            with open(self.obstacles_file, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 4):
                    if i + 3 >= len(lines):
                        break

                    obstacle = []
                    valid_corner = True
                    for j in range(4):
                        try:
                            # Split on comma and strip whitespace
                            coords = lines[i+j].strip().split(',')
                            if len(coords) != 2:
                                print(
                                    f"Invalid format at line {i+j}: {lines[i+j].strip()}")
                                valid_corner = False
                                break

                            x, y = float(coords[0].strip()), float(
                                coords[1].strip())
                            obstacle.append([x, y])
                        except (ValueError, IndexError) as e:
                            print(f"Error at line {i+j}: {e}")
                            valid_corner = False
                            break

                    if valid_corner:
                        obstacles.append(obstacle)

            if not obstacles:
                print("No valid obstacles found in file")
                return None, None, None

            print(f"Successfully loaded {len(obstacles)} obstacles")

            # Select target obstacle
            if self.target_obstacle_idx >= len(obstacles):
                print(
                    f"Warning: Target obstacle index {self.target_obstacle_idx} out of range. Using first obstacle.")
                self.target_obstacle_idx = 0
            target_obstacle = obstacles[self.target_obstacle_idx]

            # Load initial positions
            init_positions = []
            with open(self.init_pos_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            coords = line.strip().split(',')
                            x, y = float(coords[0].strip()), float(
                                coords[1].strip())
                            init_positions.append([x, y])
                        except (ValueError, IndexError) as e:
                            print(
                                f"Error parsing initial position: {line.strip()}: {e}")

            # Load goals
            goals = []
            with open(self.goals_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        try:
                            coords = line.strip().split(',')
                            x, y = float(coords[0].strip()), float(
                                coords[1].strip())
                            goals.append([x, y])
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing goal: {line.strip()}: {e}")

            print(
                f"Loaded {len(init_positions)} initial positions and {len(goals)} goals")
            return target_obstacle, init_positions, goals

        except Exception as e:
            print(f"Error loading environment: {e}")
            return None, None, None

    def initialize_orca(self):
        """Initialize ORCA simulation"""
        try:
            self.orca_instance = orca_module.init_orca()
            return True
        except Exception as e:
            print(f"Error initializing ORCA: {e}")
            return False

    def log_positions(self, time, positions, is_benign=True):
        """Log positions to appropriate file with 0.01s intervals"""
        try:
            # Determine output file
            output_file = self.benign_log if is_benign else self.attack_log

            # Format: time, x1,y1, x2,y2, x3,y3, ...
            # Time with 2 decimal places for 0.01s precision
            row = [f"{time:.2f}"]

            # Add each robot's x,y coordinates
            for pos in positions:
                row.append(f"{pos[0]:.12f}")  # x coordinate
                row.append(f"{pos[1]:.12f}")  # y coordinate

            # Write to file
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            print(f"Error logging positions: {e}")

    def write_swarmfuzz_data(self, positions_log):
        """Write data files needed by SwarmFuzz"""
        try:
            tmp_dir = os.path.join(self.fuzz_dir, 'tmp_files')
            os.makedirs(tmp_dir, exist_ok=True)

            # Write position data
            pos_csv = os.path.join(tmp_dir, f'attack_pos{self.seed_start}.csv')
            with open(pos_csv, 'w', newline='') as f:
                writer = csv.writer(f)

                for time, positions in positions_log:
                    # Format exactly as compute_vel_vasarhelyi.m:
                    # [time, x1, y1, x2, y2, x3, y3]
                    row = [f"{time:.2f}"]
                    for pos in positions:
                        row.append(f"{float(pos[0]):.12f}")  # x coordinate
                        row.append(f"{float(pos[1]):.12f}")  # y coordinate
                    writer.writerow(row)

            # Write distance data
            dist_csv = os.path.join(tmp_dir, f'dist_obs{self.seed_start}.csv')
            with open(dist_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'robot_id', 'distance'])
                for time, positions in positions_log:
                    for i, pos in enumerate(positions):
                        dx = pos[0] - 15.0
                        dy = pos[1] - 15.0
                        dist = (dx*dx + dy*dy)**0.5
                        writer.writerow([f"{time:.2f}", i+1, f"{dist:.12f}"])

            # Write collision data
            col_csv = os.path.join(tmp_dir, f'nb_col{self.seed_start}.csv')
            with open(col_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                for time, _ in positions_log:
                    writer.writerow([f"{time:.2f}", "0"])

            # Write direction data (empty file)
            dire_csv = os.path.join(tmp_dir, f'dire{self.seed_start}.csv')
            open(dire_csv, 'w').close()

            print("Data files written successfully")
            return True

        except Exception as e:
            print(f"Error writing SwarmFuzz data: {e}")
            return False

    def run_orca_benign(self):
        """Run ORCA without attacks"""
        try:
            print("Starting benign run...")
            if not self.initialize_orca():
                return False

            # Store positions for SwarmFuzz
            positions_log = []

            # Use 0.01s intervals
            total_time = 50.0  # 50 seconds total
            time_step = 0.01   # 0.01 second intervals

            print(f"Running benign simulation for {total_time:.1f} seconds...")

            counter = 0
            while counter <= total_time:
                positions = orca_module.get_positions(self.orca_instance)
                velocities = orca_module.get_velocities(self.orca_instance)

                # Store positions with timestamp
                positions_log.append((counter, positions))

                # Log positions
                self.log_positions(counter, positions, is_benign=True)

                # Convert positions and velocities to tuples
                pos_tuples = [(float(p[0]), float(p[1])) for p in positions]
                vel_tuples = [(float(v[0]), float(v[1])) for v in velocities]

                # Use the real position of the first agent
                real_pos = pos_tuples[0] if pos_tuples else (0.0, 0.0)

                # Calculate next step
                orca_module.calculate_next_positions(
                    time_step=float(time_step),
                    positions=pos_tuples,
                    velocities=vel_tuples,
                    spoofed_agent_index=int(-1),
                    spoofed_agent_position=real_pos,
                    orca_instance=self.orca_instance
                )

                counter += time_step

            print(f"Benign run completed with {len(positions_log)} timesteps")

            # Write data for SwarmFuzz
            if not self.write_swarmfuzz_data(positions_log):
                return False

            return True

        except Exception as e:
            print(f"Error in benign run: {e}")
            return False

    def prepare_swarmfuzz_files(self, target_obstacle, init_positions):
        """Prepare CSV files for SwarmFuzz"""
        try:
            # Calculate obstacle center
            center_x = sum(p[0] for p in target_obstacle) / 4
            center_y = sum(p[1] for p in target_obstacle) / 4

            # Create position CSV
            positions = np.array(init_positions)
            np.savetxt(os.path.join(self.fuzz_dir, 'attack_pos.csv'),
                       positions, delimiter=',')

            # Create distance CSV
            distances = []
            for pos in init_positions:
                dist = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                distances.append(dist)
            np.savetxt(os.path.join(self.fuzz_dir, 'dist_obs.csv'),
                       distances, delimiter=',')

            # Initialize collision CSV
            np.savetxt(os.path.join(self.fuzz_dir, 'nb_col.csv'),
                       np.zeros(len(init_positions)), delimiter=',')

            # Initialize info CSV
            np.savetxt(os.path.join(self.fuzz_dir, 'col_info.csv'),
                       np.zeros((1, 4)), delimiter=',')

            return True

        except Exception as e:
            print(f"Error preparing SwarmFuzz files: {e}")
            return False

    def run_swarmfuzz_internal(self):
        """Run SwarmFuzz internal components"""
        try:
            print("\nRunning SwarmFuzz components...")

            # Call swarmfuzz.m directly with parameters
            self.eng.swarmfuzz(
                float(self.seed_start),  # start seed
                float(self.seed_end),    # end seed
                float(self.nb_robots),   # number of robots
                float(self.dev),         # deviation
                nargout=0
            )

            print("SwarmFuzz components completed")
            return True

        except Exception as e:
            print(f"Error in SwarmFuzz internal execution: {e}")
            return False

    def apply_attack(self):
        """Apply generated attack parameters to ORCA"""
        try:
            print("Applying attack...")
            if not self.initialize_orca():
                return False

            # Clear previous log
            if os.path.exists(self.attack_log):
                os.remove(self.attack_log)

            # Read attack parameters
            attack_params = np.loadtxt(os.path.join(
                self.fuzz_dir, f'attack_param_{self.seed_start}.csv'), delimiter=',')
            start_time = attack_params[0]
            duration = attack_params[1]
            deviation = attack_params[2]

            counter = 0
            attacked_robot_id = int(self.config.get('attackedRobotId', 1))

            while counter < float(self.config.get('totalTimeStep', 50.0)):
                positions = orca_module.get_positions(self.orca_instance)
                velocities = orca_module.get_velocities(self.orca_instance)

                # Store original position
                real_pos = positions[attacked_robot_id]

                # Apply attack if within window
                if counter >= start_time and counter < (start_time + duration):
                    spoofed_pos = list(positions[attacked_robot_id])
                    spoofed_pos[1] += deviation
                    positions[attacked_robot_id] = tuple(spoofed_pos)

                # Log positions
                self.log_positions(counter, positions, is_benign=False)

                # Calculate next step
                orca_module.calculate_next_positions(
                    self.time_step,
                    positions,
                    velocities,
                    attacked_robot_id,
                    real_pos,
                    self.orca_instance
                )

                counter += self.time_step

            print("Attack application completed")
            return True

        except Exception as e:
            print(f"Error applying attack: {e}")
            return False

    def run_complete_workflow(self):
        """Run complete workflow"""
        try:
            print("Starting complete workflow...")

            # 1. Run benign case
            if not self.run_orca_benign():
                return False

            # 2. Load environment and prepare files
            target_obstacle, init_positions, goals = self.load_environment()
            if target_obstacle is None:
                return False

            if not self.prepare_swarmfuzz_files(target_obstacle, init_positions):
                return False

            # 3. Run SwarmFuzz internal components
            if not self.run_swarmfuzz_internal():
                return False

            # 4. Apply attack
            if not self.apply_attack():
                return False

            print("Complete workflow finished successfully")
            return True

        except Exception as e:
            print(f"Error in workflow: {e}")
            return False


def main():
    wrapper = OrcaSwarmFuzzWrapper(target_obstacle_idx=1)
    success = wrapper.run_complete_workflow()
    if success:
        print("Execution completed successfully")
    else:
        print("Execution failed")


if __name__ == "__main__":
    main()
