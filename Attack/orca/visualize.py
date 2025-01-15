import argparse
import yaml
from tkinter import ttk
import tkinter as tk
import numpy as np
from io import StringIO
from configparser import ConfigParser
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
import matplotlib


coll = 0


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


def get_data(filename, robot_positions):
    max_x = 0.0
    max_y = 0.0
    with open(filename) as f:
        for line in f:
            line_split = line.strip().split(" ")
            # print("Time: ", line_split[0])
            positions_at_t = []
            for pos in line_split[1:]:
                position = pos.split(',')
                # max_x and max_y only used for figure sizes
                max_x = max(max_x, float(position[0][1:]))
                max_y = max(max_y, float(position[1][:-1]))
                positions_at_t.append(
                    np.array([float(position[0][1:]), float(position[1][:-1])]))
            robot_positions.append(positions_at_t)
    # print("Max x: ", max_x, " Max y: ", max_y)


def point_line_distance(point, line_start, line_end):
    # Convert points to numpy arrays for vector operations
    point = np.array(point)
    line_start = np.array(line_start)
    line_end = np.array(line_end)

    # Calculate the vector from line_start to point and line_start to line_end
    line_vector = line_end - line_start
    point_vector = point - line_start

    # Calculate the projection
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


def visualize2D(obstacles, robots, attackers, radius, initial_positions, goals, attack_type, num_robots, writer=None):
    if writer:
        fig = plt.gcf()
        ax = fig.gca()
    else:
        fig, ax = plt.subplots(figsize=(10, 10))

    global coll
    x_coor_x = -10
    x_coor_y = 50
    y_coor_x = -10
    y_coor_y = 50
    if num_robots == 5:
        x_coor_x = -50
        x_coor_y = 350
        y_coor_x = -50
        y_coor_y = 350

    plt.xlim([x_coor_x, x_coor_y])
    plt.ylim([y_coor_x, y_coor_y])

    colors = ['red', 'blue', 'green', 'purple', 'cyan', 'magenta',
              'yellow', 'orange']
    print("Number of robots: ", num_robots)
    if num_robots > len(colors):
        raise ValueError("Not enough colors for the number of robots")

    visited_positions = {i: [] for i in range(num_robots)}

    for obstacle in obstacles:
        ax.add_patch(Polygon(obstacle, color='k', zorder=10))
        if writer:  # Only draw and pause if recording
            plt.draw()
            plt.pause(0.0000001)

    print("Robot initial positions: ", robots[0])
    # every time step
    for time_index, robot_pos_list in enumerate(robots):
        ax.cla()
        plt.xlim([x_coor_x, x_coor_y])
        plt.ylim([y_coor_x, y_coor_y])

        # Plot all visited positions for each robot
        for rob_ind, pos_list in visited_positions.items():
            if len(pos_list) > 0:
                x, y = zip(*pos_list)
                normalized = float((time_index * 10) %
                                   len(robots)) / len(robots)
                plt.scatter(x, y, marker='.',
                            color=colors[rob_ind % len(colors)], s=10, alpha=normalized, zorder=10)

        # position of the robots at time t
        for rob_ind, pos in enumerate(robot_pos_list):
            visited_positions[rob_ind].append(pos)

        for obstacle in obstacles:
            ax.add_patch(Polygon(obstacle, color='grey', zorder=10))

        # Plot initial positions and goals
        for pos in initial_positions:
            plt.scatter(pos[0], pos[1], marker='x',
                        color='orange', s=100, zorder=20)
        for goal in goals:
            plt.scatter(goal[0], goal[1], marker='x',
                        color='green', s=100, zorder=20)

        if attack_type == 5 or attack_type == 6:
            plt.scatter(-61, -40, marker='+',
                        color='red', s=100, zorder=20)

        ax = plt.gca()
        plt.xlim([x_coor_x, x_coor_y])
        plt.ylim([y_coor_x, y_coor_y])

        collision_indexes = set()
        # Check robot-robot collisions
        for i in range(len(robot_pos_list)):
            for j in range(i + 1, len(robot_pos_list)):
                # Calculate distance
                dist = np.linalg.norm(robot_pos_list[i] - robot_pos_list[j])
                if dist < 2 * radius * 0.999:
                    coll += 1
                    collision_indexes.add(i)
                    collision_indexes.add(j)
                    print(
                        f"Time {time_index}: Robot-Robot collision between robots {i} and {j} (distance: {dist:.4f})")

        # Check robot-obstacle collisions
        for i in range(len(robot_pos_list)):
            robot_pos = robot_pos_list[i]
            for obs_idx, obstacle in enumerate(obstacles):
                # Check distance to each edge of the obstacle
                for j in range(4):
                    p1 = obstacle[j]
                    p2 = obstacle[(j + 1) % 4]  # Wrap around to first point
                    dist = point_line_distance(robot_pos, p1, p2)
                    if dist < radius * 0.999:  # Just using robot radius
                        coll += 1
                        collision_indexes.add(i)
                        print(
                            f"Time {time_index}: Robot-Obstacle collision: Robot {i} with Obstacle {obs_idx} (distance: {dist:.4f}, radius: {radius})")
                        print(
                            f"    Robot position: ({robot_pos[0]:.4f}, {robot_pos[1]:.4f})")
                        print(
                            f"    Obstacle edge: ({p1[0]:.4f}, {p1[1]:.4f}) to ({p2[0]:.4f}, {p2[1]:.4f})")
                        break

        if collision_indexes:
            print(
                f"Time {time_index}: All colliding robots: {sorted(list(collision_indexes))}\n")

        # Remove or comment out this line:
        # print("Collision indexes: ", collision_indexes)

        for j in range(len(robot_pos_list)):
            edge_color = "red" if j in collision_indexes else "blue"
            circle = Circle(
                robot_pos_list[j], radius, fill=False, edgecolor=edge_color, zorder=20)
            ax.add_patch(circle)
            ax.annotate(str(j), (robot_pos_list[j][0], robot_pos_list[j][1]), color="white",
                        ha='center', va='center', fontsize=10, zorder=30)

        if writer:
            writer.grab_frame()
        else:
            plt.draw()
            plt.pause(0.01)

    if not writer:
        plt.show()


def parse_yaml_obstacle_file(file_name):
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)

    obstacles = data.get('map', {}).get('obstacles', [])
    all_obstacles = []

    for point in obstacles:
        x, y = point
        obstacle = [
            [x, y],
            [x + 1, y],
            [x + 1, y + 1],
            [x, y + 1]
        ]
        all_obstacles.append(obstacle)

    return all_obstacles


def parse_obstacle_centers_glas(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    obstacles = data.get('map', {}).get('obstacles', [])
    centers = []
    radius = 0.5

    for point in obstacles:
        x, y = point
        center_x = x + radius
        center_y = y + radius
        centers.append((center_x, center_y))

    return centers


def parse_txt_obstacle_file(file_name):
    all_obstacles = []
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Split based on the comma
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])

                    obstacle = [
                        [x - 0.5, y - 0.5],
                        [x + 0.5, y - 0.5],
                        [x + 0.5, y + 0.5],
                        [x - 0.5, y + 0.5],
                    ]

                    all_obstacles.append(obstacle)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")
    return all_obstacles


def parse_obstacle_file(file_name):
    """
    Parse obstacle file where every four lines represent one obstacle's corners.
    Returns list of obstacles, where each obstacle is a numpy array of its corners.
    """
    obstacles = []
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            # Process every 4 lines as one obstacle
            for i in range(0, len(lines), 4):
                if i + 3 >= len(lines):
                    break

                # Get the 4 corners of this obstacle
                corners = []
                for j in range(4):
                    # Strip whitespace and split on comma
                    x, y = map(float, lines[i + j].strip().split(','))
                    corners.append([x, y])

                # Convert to numpy array and add to obstacles list
                obstacles.append(np.array(corners))

        print(f"Loaded {len(obstacles)} obstacles from {file_name}")
        return obstacles

    except Exception as e:
        print(f"Error reading obstacle file {file_name}: {e}")
        return []


obstacles2 = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true',
                        help='Record animation to video')
    args = parser.parse_args()

    fig = plt.figure(figsize=(10, 10))  # Create only one figure

    global obstacles2
    robot_positions = []

    # robot positions: shows all robot locations at time t
    environment_file = None
    config = parse_config(
        "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")
    num_robots = int(config.get("num_robots", 0))
    log_file = config.get("log_file", "")
    time_step = float(config.get("time_step", 0.0))
    environment_file = config.get("environment_file", "")
    neighbor_dist = float(config.get("neighborDist", 0.0))
    max_neighbors = int(config.get("maxNeighbors", 0))
    time_horizon = float(config.get("timeHorizon", 0.0))
    time_horizon_obst = float(config.get("timeHorizonObst", 0.0))
    radius = float(config.get("radius", 0.0))
    max_speed = float(config.get("maxSpeed", 0.0))
    attack_type = int(config.get("attackType", 0))

    # Read initial positions and goals
    initial_positions = np.loadtxt(
        environment_file + "robot_pos.txt", delimiter=',')
    goals = np.loadtxt(environment_file + "goals.txt", delimiter=',')

    get_data(log_file, robot_positions)
    print("One robot position as sample: ",
          robot_positions[0][0], type(robot_positions[0][0]))
    print("Size of robot position changes: ", len(robot_positions))
    print("radius: ", radius)

    # Obstacles map construction
    """
    obstacles2 = [
        # rectangle
        np.array([[-75, -25], [-65, -25], [-75, 25], [-65, 25]])
        ]
    obstacles = [

        # angle
        # bottom line
        np.array([[5, 0], [25, 0], [25, 3], [5, 3]]),
        # left line
        np.array([[5, 3], [8, 3], [8, 15], [5, 15]]),

        # angle
        # bottom line
        np.array([[-20, -2], [-5, -2], [-5, -18], [-2, -18]]),
        # right line
        np.array([[-7, -18], [-5, -8], [-5, -8], [-7, -8]]),

        # walls
        # bottom line
        np.array([[-100, -100], [100, -100], [100, -99.7], [-100, -99.7]]),  # comment this for better 3D visualization
        # top line
        np.array([[-100, 99.7], [100, 99.7], [100, 100], [-100, 100]]),
        # left line
        np.array([[-100, -99.7], [-99.7, -99.7], [-99.7, 99.7], [-100, 99.7]]),
        # right line
        np.array([[99.7, -99.7], [100, -99.7], [100, 99.7], [99.7, 99.7]]),  # comment this for better 3D visualization

        # moving obstacle
        np.array([[-2.3, 2.0], [-2.2, 2.0], [-2.2, 2.1], [-2.3, 2.1]]),
        np.array([[2.3, -2.3], [2.4, -2.3], [2.4, -2.2], [2.3, -2.2]]),
        np.array([[0.0, -2.3], [0.1, -2.3], [0.1, -2.2], [0.0, -2.2]]),
    ]

    obstacles_from_block = [
        np.array([[-10, 40], [-40, 40], [-40, 10], [-10, 10]]),
        np.array([[10, 40], [40, 40], [40, 10], [10, 10]]),
        np.array([[10, -40], [40, -40], [40, -10], [10, -10]]),
        np.array([[-10, -40], [-10, -10], [-40, -10], [-40, -40]]),
    ]
    """
    obstacles = parse_obstacle_file(environment_file + "obstacles.txt")
    # obstacles2 = parse_obstacle_centers_glas(
    #     "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
    # # obstacles = parse_yaml_obstacle_file(
    #     "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
    print("Obstacles: ", obstacles)
    # obstacles = [
    #    np.array([[-10, 40], [-40, 40], [-40, 10], [-10, 10]]),
    #    np.array([[10, 40], [10, 10], [40, 10], [40, 40]]),
    #    np.array([[10, -40], [40, -40], [40, -10], [10, -10]]),
    #    np.array([[-10, -40], [-10, -10], [-40, -10], [-40, -40]]),
    # ]

    U_shape_obstacle = []
    # U_shape_obstacle = [
    #    np.array([[-40, 40], [-40, -40], [40, -40], [40, 40],
    #    [30,40],[30,-30],[-30,-30],[-30,40]])
    # ]

    if args.record:
        writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        fig = plt.figure(figsize=(10, 10))
        with writer.saving(fig, "robot_visualization.mp4", 100):
            visualize2D(obstacles=obstacles, robots=robot_positions,
                        attackers=[], radius=radius,
                        initial_positions=initial_positions, goals=goals,
                        attack_type=attack_type, num_robots=num_robots,
                        writer=writer)
    else:
        visualize2D(obstacles=obstacles, robots=robot_positions,
                    attackers=[], radius=radius,
                    initial_positions=initial_positions, goals=goals,
                    attack_type=attack_type, num_robots=num_robots,
                    writer=None)
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    main()
