import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FFMpegWriter


def parse_config(filename):
    """Parse configuration file."""
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip().strip('"')
    return config


def load_robot_positions(filename):
    """Load robot positions from log file."""
    positions = []
    with open(filename) as f:
        for line in f:
            time_step = []
            tokens = line.strip().split()
            time = float(tokens[0])
            for pos in tokens[1:]:
                x, y = map(float, pos.strip('()').split(','))
                time_step.append(np.array([x, y]))
            positions.append(time_step)
    return positions


def load_environment(env_path):
    """Load environment data (obstacles, initial positions, goals)."""
    obstacles = np.loadtxt(f"{env_path}/obstacles.txt",
                           delimiter=',').reshape(-1, 4, 2)
    initial_positions = np.loadtxt(f"{env_path}/robot_pos.txt", delimiter=',')
    goals = np.loadtxt(f"{env_path}/goals.txt", delimiter=',')
    return obstacles, initial_positions, goals


def check_collisions(robot_positions, obstacles, radius):
    """Check for robot-robot and robot-obstacle collisions."""
    collision_indexes = set()

    # Robot-robot collisions
    for i in range(len(robot_positions)):
        for j in range(i + 1, len(robot_positions)):
            dist = np.linalg.norm(robot_positions[i] - robot_positions[j])
            if dist < 2 * radius * 0.999:
                collision_indexes.update([i, j])

    # Robot-obstacle collisions
    for i, robot_pos in enumerate(robot_positions):
        for obstacle in obstacles:
            for j in range(4):
                p1, p2 = obstacle[j], obstacle[(j + 1) % 4]
                dist = point_line_distance(robot_pos, p1, p2)
                if dist < radius * 0.999:
                    collision_indexes.add(i)
                    break

    return collision_indexes


def point_line_distance(point, line_start, line_end):
    """Calculate distance from point to line segment."""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.dot(line_vec, line_vec)

    if line_len == 0:
        return np.linalg.norm(point_vec)

    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def visualize(robot_positions, obstacles, initial_positions, goals, radius, num_robots, save_video=False):
    """Visualize robot movements."""
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'purple',
              'cyan', 'magenta', 'yellow', 'orange']

    # Setup video writer if saving
    writer = None
    if save_video:
        writer = FFMpegWriter(fps=10, metadata=dict(artist='ORCA Visualizer'))
        writer.setup(fig, "robot_movement.mp4", dpi=100)

    # Track robot paths
    paths = {i: [] for i in range(num_robots)}

    for time_step, positions in enumerate(robot_positions):
        ax.clear()

        # Set plot limits to -5 to 5 range
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        # Draw grid lines
        ax.grid(True, linestyle='--', alpha=0.3)

        # Draw obstacles
        for obstacle in obstacles:
            ax.add_patch(Polygon(obstacle, color='gray', alpha=0.5))

        # Draw initial positions and goals
        ax.scatter(initial_positions[:, 0], initial_positions[:, 1],
                   marker='x', color='orange', s=100, label='Start')
        ax.scatter(goals[:, 0], goals[:, 1],
                   marker='*', color='green', s=100, label='Goal')

        # Update and draw robot paths
        collision_indexes = check_collisions(positions, obstacles, radius)

        for i, pos in enumerate(positions):
            # Update path
            paths[i].append(pos)
            path = np.array(paths[i])

            # Draw path
            ax.plot(path[:, 0], path[:, 1],
                    color=colors[i % len(colors)],
                    alpha=0.3,
                    linewidth=1)

            # Draw robot
            color = 'red' if i in collision_indexes else colors[i % len(
                colors)]
            circle = Circle(pos, radius, fill=False, color=color, linewidth=2)
            ax.add_patch(circle)

            # Add robot label
            ax.annotate(str(i), pos, color='black',
                        ha='center', va='center')

        ax.legend()
        ax.set_title(f'Time Step: {time_step}')

        if writer:
            writer.grab_frame()
        else:
            plt.pause(0.05)

    if writer:
        writer.finish()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='ORCA Robot Visualization')
    parser.add_argument('--save-video', action='store_true',
                        help='Save animation as video')
    args = parser.parse_args()

    # Use specific config file path
    config_path = "../False_Data_Attacks_on_ORCA/Attack/src/config.txt"

    # Load configuration
    config = parse_config(config_path)

    # Get parameters from config
    num_robots = int(config['num_robots'])
    radius = float(config['radius'])
    log_file = config['log_file']
    env_path = config['environment_file']

    print(f"Loading configuration from: {config_path}")
    print(f"Number of robots: {num_robots}")
    print(f"Robot radius: {radius}")
    print(f"Log file: {log_file}")
    print(f"Environment path: {env_path}")

    # Load data
    robot_positions = load_robot_positions(log_file)
    obstacles, initial_positions, goals = load_environment(env_path)

    # Run visualization
    visualize(robot_positions, obstacles, initial_positions, goals,
              radius, num_robots, args.save_video)


if __name__ == '__main__':
    main()
