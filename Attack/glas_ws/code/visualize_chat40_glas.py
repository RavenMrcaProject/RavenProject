from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np
import yaml
import math
import argparse
from matplotlib.animation import FFMpegWriter

coll = 0


def parse_config(filename):
    config = {}
    with open(filename, 'r') as file:
        key, value = None, ""
        for line in file:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith('"') and key is not None:
                value += line.strip('"')
                config[key] = value
                key, value = None, ""
                continue
            parts = line.split('=', 1)
            if len(parts) != 2:
                print(f"Skipping invalid line: {line}")
                continue
            key, value = parts[0].strip(), parts[1].strip().strip('"')
            if value.endswith("\\"):
                value = value[:-1].strip('"')
            else:
                # Convert numeric values
                if key in ['radius', 'time_step', 'neighborDist', 'maxNeighbors',
                           'timeHorizon', 'timeHorizonObst', 'maxSpeed', 'num_robots']:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        print(
                            f"Warning: Could not convert {key}={value} to float")
                        config[key] = value
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
            positions_at_t = []
            for pos in line_split[1:]:
                position = pos.split(',')
                max_x = max(max_x, float(position[0][1:]))
                max_y = max(max_y, float(position[1][:-1]))
                positions_at_t.append(
                    np.array([float(position[0][1:]), float(position[1][:-1])]))
            robot_positions.append(positions_at_t)


def point_to_box_distance(point, box_corners):
    # Calculate the closest point on the box to the given point
    x = max(min(point[0], box_corners[1][0]), box_corners[0][0])
    y = max(min(point[1], box_corners[2][1]), box_corners[0][1])
    closest_point = np.array([x, y])
    return np.linalg.norm(point - closest_point)


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


def visualize2D(obstacles, robots, attackers, radius, initial_positions, goals, attack_type, num_robots, record=False, writer=None, fig=None, ax=None):
    global coll
    print(f"\nDebug - Starting visualization")
    print(f"Number of timesteps: {len(robots)}")
    print(f"Number of obstacles: {len(obstacles)}")
    print(f"Victim Robot ID: {victimRobotId}")
    print(f"Robot radius: {radius}")

    if not record:
        fig, ax = plt.subplots(figsize=(10, 10))

    x_coor_x, x_coor_y = -10, 50
    y_coor_x, y_coor_y = -10, 50
    if num_robots == 5:
        x_coor_x, x_coor_y = -50, 350
        y_coor_x, y_coor_y = -50, 350

    colors = ['red', 'blue', 'green', 'purple',
              'cyan', 'magenta', 'yellow', 'orange']
    visited_positions = {i: [] for i in range(num_robots)}

    for time_index, robot_pos_list in enumerate(robots):
        ax.cla()
        ax.set_xlim([x_coor_x, x_coor_y])
        ax.set_ylim([y_coor_x, y_coor_y])

        # Draw trails with alpha values
        for rob_ind, pos_list in visited_positions.items():
            if len(pos_list) > 0:
                x, y = zip(*pos_list)
                normalized = float((time_index * 10) %
                                   len(robots)) / len(robots)
                ax.scatter(x, y, marker='.', color=colors[rob_ind % len(colors)],
                           s=10, alpha=normalized, zorder=10)

        # Update visited positions
        for rob_ind, pos in enumerate(robot_pos_list):
            visited_positions[rob_ind].append(pos)

        # Draw obstacles
        for obstacle in obstacles:
            ax.add_patch(Polygon(obstacle, color='grey', zorder=10))

        # Draw initial positions and goals
        for pos in initial_positions:
            ax.scatter(pos[0], pos[1], marker='x',
                       color='orange', s=100, zorder=20)
        for goal in goals:
            ax.scatter(goal[0], goal[1], marker='x',
                       color='green', s=100, zorder=20)

        collision_indexes = set()

        # Robot-robot collision detection
        for i in range(len(robot_pos_list)):
            for j in range(i + 1, len(robot_pos_list)):
                dist = np.linalg.norm(robot_pos_list[i] - robot_pos_list[j])
                if dist < 2 * radius * 0.999:
                    coll += 1
                    collision_indexes.add(i)
                    collision_indexes.add(j)
                    print(
                        f"Time {time_index}: Robot-Robot collision between robots {i} and {j} (distance: {dist:.4f})")
                    if i == victimRobotId or j == victimRobotId:
                        print(
                            f"  WARNING: Collision involves victim robot {victimRobotId}")

        # Robot-obstacle collision detection
        for i in range(len(robot_pos_list)):
            robot_pos = robot_pos_list[i]
            for obs_idx, obstacle in enumerate(obstacles):
                for j in range(len(obstacle)):
                    p1 = obstacle[j]
                    p2 = obstacle[(j + 1) % len(obstacle)]
                    dist = point_line_distance(robot_pos, p1, p2)
                    if dist < radius * 0.999:
                        coll += 1
                        collision_indexes.add(i)
                        print(
                            f"Time {time_index}: Robot-Obstacle collision: Robot {i} with Obstacle {obs_idx}")
                        print(f"    Distance: {dist:.4f}, Radius: {radius}")
                        print(
                            f"    Robot position: ({robot_pos[0]:.4f}, {robot_pos[1]:.4f})")
                        if i == victimRobotId:
                            print(
                                f"    WARNING: Collision involves victim robot {victimRobotId}")
                        break

        # Draw robots
        for j in range(len(robot_pos_list)):
            edge_color = "red" if j in collision_indexes else "blue"
            circle = Circle(
                robot_pos_list[j], radius, fill=False, edgecolor=edge_color, zorder=20)
            ax.add_patch(circle)
            ax.annotate(str(j), (robot_pos_list[j][0], robot_pos_list[j][1]),
                        color="white", ha='center', va='center', fontsize=10, zorder=30)

        if record:
            writer.grab_frame()
        else:
            plt.draw()
            plt.pause(0.01)

    print(f"\nTotal number of collisions: {coll}")


def parse_yaml_obstacle_file(file_name):
    """Parse GLAS yaml file for obstacle corners."""
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)
    obstacles = data.get('map', {}).get('obstacles', [])
    all_obstacles = []
    for point in obstacles:
        x, y = point
        obstacle = [
            [x, y],           # bottom-left
            [x + 1, y],       # bottom-right
            [x + 1, y + 1],   # top-right
            [x, y + 1]        # top-left
        ]
        all_obstacles.append(obstacle)
    return all_obstacles


def parse_yaml_goals(file_name):
    """Parse GLAS yaml file for goal positions."""
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)
    agents = data.get('agents', [])
    return [agent.get('goal') for agent in agents]


def parse_yaml_initial_positions(file_name):
    """Parse GLAS yaml file for initial positions."""
    with open(file_name, 'r') as file:
        data = yaml.safe_load(file)
    agents = data.get('agents', [])
    return [agent.get('start') for agent in agents]


def parse_txt_initial_positions(file_name):
    """Parse ORCA txt file for initial positions."""
    initial_positions = []
    with open(file_name, 'r') as file:
        for line in file:
            position = list(map(float, line.strip().split(',')))
            initial_positions.append(position)
    return initial_positions


def parse_txt_goals(file_name):
    """Parse ORCA txt file for goals."""
    goals = []
    with open(file_name, 'r') as file:
        for line in file:
            goal = list(map(float, line.strip().split(',')))
            goals.append(goal)
    return goals


def parse_txt_obstacles(file_name):
    """Parse ORCA txt file for obstacles."""
    obstacles = []
    with open(file_name, 'r') as file:
        for line in file:
            x, y, _ = map(int, line.strip().split(','))
            obstacle = [
                [x - 1, y - 1],  # bottom-left
                [x + 1, y - 1],  # bottom-right
                [x + 1, y + 1],  # top-right
                [x - 1, y + 1]   # top-left
            ]
            obstacles.append(obstacle)
    return obstacles


def parse_obstacle_centers_orca(file_path):
    """Parse ORCA obstacle centers file."""
    with open(file_path, 'r') as file:
        return [tuple(map(float, line.strip().split(','))) for line in file]


def parse_obstacle_centers_glas(file_path):
    """Parse GLAS yaml file for obstacle centers."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    obstacles = data.get('map', {}).get('obstacles', [])
    return [np.array([x + 0.5, y + 0.5]) for x, y in obstacles]


# temp values
obstacles2 = []
victimRobotId = -1


def main():
    parser = argparse.ArgumentParser(description='Visualize robot movements')
    parser.add_argument('--record', action='store_true',
                        help='Record animation to MP4')
    parser.add_argument('--format', choices=['glas', 'orca'], default='glas',
                        help='Format of input files (default: glas)')
    args = parser.parse_args()

    global obstacles2, victimRobotId
    robot_positions = []

    # Parse configuration
    # TODO: change the path to the config file
    config = parse_config(
        "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")
    num_robots = int(config.get("num_robots", 0))
    log_file = config.get("log_file", "")
    time_step = float(config.get("time_step", 0))
    environment_file = config.get("environment_file", "")
    radius = float(config.get("radius", 0))
    attack_type = int(config.get("attackType", 0))
    victimRobotId = int(config.get("victimRobotId", 0))

    print("num_robots: ", num_robots)
    print("log_file: ", log_file)
    print("time_step: ", time_step)
    print("environment_file: ", environment_file)
    print("radius: ", radius)
    print("attack_type: ", attack_type)
    print("victimRobotId: ", victimRobotId)

    # glas parser
    # TODO: change the path to the config file
    initial_positions = parse_yaml_initial_positions(
        "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
    goals = parse_yaml_goals(
        "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
    # corners
    obstacles = parse_yaml_obstacle_file(
        "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
    # centers
    obstacles2 = parse_obstacle_centers_glas(
        "../comparison/swarmlab/glas_ws/results/singleintegrator/instances/map_8by8_obst12_agents8_ex0000.yaml")
    # orca parser
    # initial_positions = parse_txt_initial_positions(
    #     environment_file + "robot_pos.txt")
    # goals = parse_txt_goals(
    #     environment_file + "goals.txt")
    # obstacles = parse_txt_obstacles(
    #     environment_file + "obstacles.txt")
    # obstacles2 = parse_obstacle_centers_orca(
    #     environment_file + "obstacles_2.txt")
    print("initial_positions: ", initial_positions)
    print("goals: ", goals)
    print("obstacles: ", obstacles)
    print("obstacles2: ", obstacles2)

    # Add debug print after loading obstacles
    print("Debug - First few obstacle centers:")
    for i, obs in enumerate(obstacles2[:3]):
        print(f"Obstacle {i}: {obs}")

    get_data(log_file, robot_positions)

    # Setup visualization
    if args.record:
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        fig, ax = plt.subplots(figsize=(10, 10))
        with writer.saving(fig, "robot_visualization.mp4", 100):
            visualize2D(obstacles=obstacles, robots=robot_positions, attackers=[],
                        radius=radius, initial_positions=initial_positions,
                        goals=goals, attack_type=attack_type, num_robots=num_robots,
                        record=True, writer=writer, fig=fig, ax=ax)
    else:
        visualize2D(obstacles=obstacles, robots=robot_positions, attackers=[],
                    radius=radius, initial_positions=initial_positions,
                    goals=goals, attack_type=attack_type, num_robots=num_robots,
                    record=False, writer=None, fig=None, ax=None)

    plt.show()


if __name__ == '__main__':
    main()
