import orca_module
import random
from configparser import ConfigParser


def parse_config(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            # Remove whitespace and skip comments
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Split on the first equals sign
            parts = line.split('=', 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip().strip('"')
                config[key] = value

    return config


# Read configuration
config = parse_config(
    "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")

# Get values from config
num_robots = int(config.get("num_robots", 3))  # Default to 3 if not found
time_step = float(config.get("time_step", 1.0))  # Default to 1.0 if not found
attackedRobotId = int(config.get("attackedRobotId", 1)
                      )  # Default to 1 if not found
iterations = int(float(config.get("totalTimeStep", 80))
                 )  # Convert to float first, then int

orca_instance = orca_module.init_orca()

# Example initial positions and velocities for testing
updated_velocities = None
updated_positions = None

# Open file to write the output
output_file = open('positions.txt', 'w')

# Running the ORCA algorithm
counter = None
for iteration in range(iterations):
    counter = iteration * time_step

    # Get updated velocities and positions
    updated_velocities = orca_module.get_velocities(orca_instance)
    updated_positions = orca_module.get_positions(orca_instance)

    # Write positions to file in the specified format
    output_file.write("{:.1f}".format(counter))
    for pos in updated_positions:
        # Adding a fixed z-coordinate of 5.0 to match the format
        output_file.write(" ({:.4f},{:.4f},5.0000)".format(pos[0], pos[1]))
    output_file.write("\n")

    # Calculate the next positions
    # orca_module.calculate_next_positions(
    #     time_step, updated_positions, updated_velocities, -1, updated_positions[1], orca_instance)
    positions = orca_module.get_positions(orca_instance)
    velocities = orca_module.get_velocities(orca_instance)
    real_pos = positions[attackedRobotId]
    # if counter > 5 and counter < 35:
    #     spoofed_pos = (13.45, 14.75)
    #     positions[attackedRobotId] = spoofed_pos

    orca_module.calculate_next_positions(
        time_step,
        positions,  # contains spoofed position
        velocities,
        attackedRobotId,
        real_pos,   # use real position for calculations
        orca_instance
    )

    # Rest of your existing code...
    print("\nIteration {}".format(iteration + 1))
    print("\nUpdated Velocities:")
    for idx, vel in enumerate(updated_velocities):
        print("Robot {}: Velocity = {}".format(idx + 1, vel))

    print("\nUpdated Positions:")
    for idx, pos in enumerate(updated_positions):
        print("Robot {}: Position = {}".format(idx + 1, pos))

    print("\n----------------------------------------\n")


# Close the file
output_file.close()
