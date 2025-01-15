import numpy as np
from typing import List, Tuple, Dict
import csv
import os
import orca_module
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_config(filename: str) -> Dict:
    """Parse configuration file"""
    try:
        config = {}
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip().strip('"')
                    config[key] = value
        return config
    except Exception as e:
        logging.error(f"Error reading config file: {str(e)}")
        raise


def pos_csv_update(positions: List[Tuple[float, float, float]], time: float, csv_file: str) -> None:
    """Update position CSV file: time, x1, y1, x2, y2, x3, y3, ..."""
    try:
        pos_row = [time]  # Use raw time, no multiplication
        for pos in positions:
            pos_row.extend([pos[0], pos[1]])

        logging.debug(f"Time: {time}")
        logging.debug(f"Position row: {pos_row}")

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(pos_row)

    except Exception as e:
        logging.error(f"Error updating position CSV: {str(e)}")
        logging.error(f"Positions: {positions}")
        logging.error(f"Time: {time}")
        raise


def read_obstacles(obstacle_file: str) -> List[List[Tuple[float, float]]]:
    """
    Read obstacles from file. Each obstacle is defined by 4 corners.
    Format: x,y coordinates for each corner, counterclockwise from bottom left
    """
    try:
        obstacles = []
        current_obstacle = []

        with open(obstacle_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    x, y = map(float, line.strip().split(','))
                    current_obstacle.append((x, y))

                    if len(current_obstacle) == 4:
                        obstacles.append(current_obstacle)
                        current_obstacle = []

        if not obstacles:
            raise ValueError("No valid obstacles found in file")

        logging.info(f"Read {len(obstacles)} obstacles from file")
        return obstacles

    except Exception as e:
        logging.error(f"Error reading obstacle file: {str(e)}")
        raise


def point_line_distance(point: Tuple[float, float],
                        line_start: Tuple[float, float],
                        line_end: Tuple[float, float]) -> float:
    """Calculate minimum distance from point to line segment"""
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    line_vector = line_end - line_start
    point_vector = point - line_start

    line_length_squared = np.dot(line_vector, line_vector)
    if line_length_squared == 0:
        return np.linalg.norm(point_vector)

    projection = np.dot(point_vector, line_vector) / line_length_squared

    if projection < 0:
        closest_point = line_start
    elif projection > 1:
        closest_point = line_end
    else:
        closest_point = line_start + projection * line_vector

    return np.linalg.norm(point - closest_point)


def get_min_distance_to_obstacle(position: Tuple[float, float],
                                 obstacle: List[Tuple[float, float]]) -> float:
    """Calculate minimum distance from position to any edge of the obstacle"""
    min_distance = float('inf')

    for i in range(4):
        point1 = obstacle[i]
        point2 = obstacle[(i + 1) % 4]
        distance = point_line_distance(position, point1, point2)
        min_distance = min(min_distance, distance)

    return min_distance


def dist_csv_update(positions: List[Tuple[float, float, float]],
                    obstacles: List[List[Tuple[float, float]]],
                    time: float, csv_file: str) -> None:
    """Update distance CSV file with minimum distances to obstacles"""
    try:
        for agent_id, pos in enumerate(positions):
            # Get 2D position (ignore z)
            pos_2d = (pos[0], pos[1])
            # Calculate minimum distance to obstacle
            dist = get_min_distance_to_obstacle(pos_2d, obstacles[0])

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time, agent_id + 1, dist])
    except Exception as e:
        logging.error(f"Error updating distance CSV: {str(e)}")


def col_csv_update(positions: List[Tuple[float, float, float]],
                   obstacles: List[List[Tuple[float, float]]],
                   agent_radius: float, time: float,
                   start_t: float, dur: float,
                   vic_id: int, csv_file: str) -> None:
    """Update collision CSV file checking collisions with square obstacles"""
    try:
        nb_collisions = 0
        print(f"\nChecking collisions at time {time}")
        print(f"Number of positions: {len(positions)}")
        print(f"Victim ID: {vic_id}")

        if vic_id == 0:  # Check all agents
            for i, pos in enumerate(positions):
                pos_2d = (pos[0], pos[1])
                dist = get_min_distance_to_obstacle(pos_2d, obstacles[0])
                print(
                    f"Agent {i+1}: distance to obstacle = {dist}, radius = {agent_radius}")
                if dist < agent_radius:
                    nb_collisions += 1
                    print(f"Collision detected for agent {i+1}")
        else:  # Check only victim
            if vic_id > len(positions):  # Add bounds check
                raise IndexError(
                    f"Victim ID {vic_id} is larger than number of agents {len(positions)}")
            vic_pos = positions[vic_id - 1]
            pos_2d = (vic_pos[0], vic_pos[1])
            dist = get_min_distance_to_obstacle(pos_2d, obstacles[0])
            print(
                f"Victim {vic_id}: distance to obstacle = {dist}, radius = {agent_radius}")
            if dist < agent_radius:
                nb_collisions += 1
                print("Collision detected for victim")

        print(f"Total collisions: {nb_collisions}")

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time, nb_collisions, start_t, dur])
    except Exception as e:
        logging.error(f"Error updating collision CSV: {str(e)}")
        print(f"Error details:")
        print(f"Time: {time}")
        print(f"Positions length: {len(positions)}")
        print(f"Victim ID: {vic_id}")
        raise


def info_csv_update(positions: List[Tuple[float, float, float]],
                    obstacles: List[List[Tuple[float, float]]],
                    vic_id: int, time: float, csv_file: str) -> None:
    """Update info CSV file with victim's distance to obstacle"""
    try:
        if vic_id > 0:
            vic_pos = positions[vic_id - 1]
            pos_2d = (vic_pos[0], vic_pos[1])
            dist = get_min_distance_to_obstacle(pos_2d, obstacles[0])

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time, dist])
    except Exception as e:
        logging.error(f"Error updating info CSV: {str(e)}")


def get_obstacle_center(obstacle: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate center point of obstacle from corners (for MATLAB prepare)"""
    x_sum = sum(p[0] for p in obstacle)
    y_sum = sum(p[1] for p in obstacle)
    return (x_sum/4, y_sum/4)

# TODO: change the path to the config file


def example_orca(start_t: float, dur: float, att_id: int, vic_id: int,
                 dev_y: float, seed: int, pos_csv: str, dist_csv: str,
                 col_csv: str, info_csv: str,
                 obstacle_file: str = "../False_Data_Attacks_on_ORCA/Attack/environments/env-orca-1_comparison/obstacles.txt") -> None:
    """
    Main ORCA simulation function with attack implementation
    """
    try:
        # Set random seed
        np.random.seed(seed)

        # Read configuration
        # TODO: change the path to the config file
        config = parse_config(
            "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")
        time_step = float(config.get("time_step", 0.1))
        end_time = float(config.get("totalTimeStep", 200)) * time_step

        print("\nSimulation parameters:")
        print(f"Start time: 0")
        print(f"End time: {end_time}")
        print(f"Time step: {time_step}")
        print(f"Expected steps: {int(end_time/time_step)}")

        print(f"\nMatrix dimensions:")
        print(f"Number of time steps: {int(end_time/time_step)}")
        print(f"Time range: 0 to {end_time} seconds")
        print(f"Maximum array index: {int(end_time/time_step) - 1}")

        agent_radius = float(config.get("radius", 2.0))

        # Read obstacles
        obstacles = read_obstacles(obstacle_file)

        # Initialize ORCA
        try:
            orca_instance = orca_module.init_orca()
            logging.info("ORCA initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize ORCA: {str(e)}")
            raise

        # Clear existing CSV files
        for file in [pos_csv, dist_csv, col_csv, info_csv]:
            if os.path.exists(file):
                os.remove(file)

        # Main simulation loop
        for time in np.arange(0, end_time, time_step):
            try:
                positions = orca_module.get_positions(orca_instance)
                velocities = orca_module.get_velocities(orca_instance)
                print(f"\nTime: {time}")
                print(f"Number of positions: {len(positions)}")
                print(f"Number of velocities: {len(velocities)}")
                print(f"MATLAB Attack ID (1-based): {att_id}")
                print(f"Python Attack ID (0-based): {att_id - 1}")
            except Exception as e:
                logging.error(
                    f"Error getting ORCA state at t={time}: {str(e)}")
                raise

            try:
                # Convert from MATLAB's 1-based to Python's 0-based indexing
                attack_idx = att_id - 1  # Convert to 0-based
                real_pos = positions[attack_idx]
                print(
                    f"Real position for agent {att_id} (index {attack_idx}): {real_pos}")

                # Apply attack if needed
                if time >= start_t and time < (start_t + dur) and att_id > 0:
                    positions[attack_idx] = (real_pos[0], real_pos[1] + dev_y)
                    print(f"Attack applied at t={time}: dev_y={dev_y}")

                print(f"Updating ORCA with:")
                print(f"- Positions: {positions}")
                print(f"- Velocities: {velocities}")

                orca_module.calculate_next_positions(
                    time_step,
                    positions,
                    velocities,
                    attack_idx,  # Use 0-based index
                    real_pos,
                    orca_instance
                )
                print(f"ORCA updated successfully")
            except Exception as e:
                logging.error(
                    f"Error in ORCA calculation at t={time}: {str(e)}")
                print(f"Error details:")
                print(f"Time: {time}")
                print(f"Positions: {positions}")
                print(f"Velocities: {velocities}")
                print(f"MATLAB Attack ID (1-based): {att_id}")
                print(f"Python Attack ID (0-based): {attack_idx}")
                print(f"Real position: {real_pos}")
                raise

            # Update CSV files
            pos_csv_update(positions, time, pos_csv)
            dist_csv_update(positions, obstacles, time, dist_csv)
            col_csv_update(positions, obstacles, agent_radius, time,
                           start_t, dur, vic_id, col_csv)
            info_csv_update(positions, obstacles, vic_id, time, info_csv)

        logging.info("Simulation completed successfully")

    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    try:
        example_orca(
            start_t=0, dur=0, att_id=0, vic_id=0, dev_y=0, seed=200,
            pos_csv="pos.csv", dist_csv="dist.csv",
            col_csv="col.csv", info_csv="info.csv"
        )
    except Exception as e:
        logging.error(f"Example run failed: {str(e)}")
