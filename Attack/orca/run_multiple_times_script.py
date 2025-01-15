#!/usr/bin/env python3

import subprocess
import time
import re
import sys
import os


def run_wrapper(orca_dir):
    wrapper_cmd = [
        "python3",
        os.path.join(orca_dir, "wrapper.py")
    ]
    print(f"[INFO] Running wrapper.py in: {orca_dir}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Python path: {sys.path}")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = orca_dir
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{orca_dir}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = orca_dir

        if sys.platform == 'darwin':
            if 'DYLD_LIBRARY_PATH' in env:
                env['DYLD_LIBRARY_PATH'] = f"{orca_dir}:{env['DYLD_LIBRARY_PATH']}"
            else:
                env['DYLD_LIBRARY_PATH'] = orca_dir

        print(f"[DEBUG] Environment variables:")
        print(f"PYTHONPATH: {env.get('PYTHONPATH')}")
        print(f"LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH')}")
        print(f"DYLD_LIBRARY_PATH: {env.get('DYLD_LIBRARY_PATH')}")

        subprocess.run(wrapper_cmd, cwd=orca_dir, env=env, check=True)
        print(f"[INFO] Completed wrapper.py in: {orca_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] wrapper.py failed with exit code {e.returncode}")
        return False
    return True


def run_visualize(visualize_dir, record_video=False):
    visualize_cmd = [
        "python3.8",
        "visualize.py"
    ]

    if record_video:
        visualize_cmd.append("--record")
        print("[INFO] Recording video for first iteration...")

    print(f"[INFO] Running visualize.py in: {visualize_dir}")

    # Create a new environment with MPLBACKEND set
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'  # Force the Agg backend

    proc_visual = subprocess.Popen(
        visualize_cmd,
        cwd=visualize_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # If recording, wait for completion
    if record_video:
        stdout, stderr = proc_visual.communicate()
        print(stdout)  # Show recording progress
        if proc_visual.returncode != 0:
            print(f"[ERROR] Video recording failed: {stderr}")

    return proc_visual


def check_collisions(proc_visual, timeout=10):
    collisions_found = False
    collision_indexes = []
    start_time = time.time()

    print("[INFO] Monitoring visualize.py output for collisions...")

    while True:
        # Check if there is any output
        if proc_visual.poll() is not None:
            # Process has terminated
            break

        # Read a line from stdout
        line = proc_visual.stdout.readline()
        if line:
            line_stripped = line.strip()
            # Debug: print the line
            # print(f"[DEBUG] {line_stripped}")
            # Check for non-empty collision set
            collision_match = re.search(
                r"Collision indexes:\s*\{([^}]*)\}", line_stripped)
            if collision_match:
                indexes = collision_match.group(1).strip()
                if indexes:  # Non-empty set indicates collision
                    collisions_found = True
                    collision_indexes.append(indexes)
                    # Optionally, you can break early if a collision is found
                    # break

        # Check if timeout has been reached
        if time.time() - start_time > timeout:
            break

    return collisions_found, collision_indexes


def terminate_process(proc_visual):
    if proc_visual.poll() is None:  # If process is still running
        print("[INFO] Terminating visualize.py...")
        proc_visual.terminate()
        try:
            # Wait up to 5 seconds for graceful termination
            proc_visual.wait(timeout=5)
            print("[INFO] visualize.py terminated gracefully.")
        except subprocess.TimeoutExpired:
            print(
                "[WARNING] visualize.py did not terminate in time. Killing the process...")
            proc_visual.kill()  # Force kill if terminate doesn't work
            try:
                proc_visual.wait(timeout=5)  # Wait for the kill to complete
                print("[INFO] visualize.py killed.")
            except subprocess.TimeoutExpired:
                print("[ERROR] Failed to kill visualize.py")

        # On macOS, we might need to kill any remaining Python processes
        if sys.platform == 'darwin':
            try:
                subprocess.run(['pkill', '-f', 'visualize.py'], check=False)
            except Exception as e:
                print(
                    f"[WARNING] Error while trying to kill remaining processes: {e}")


def main():
    # TODO: change the path to the config file
    orca_dir = "../comparison/swarmlab/orca"
    visualize_dir = "../False_Data_Attacks_on_ORCA/ORCASimulator"
    total_iterations = 50
    collision_count = 0
    record_first_iteration = True

    # Open log files
    collision_log_path = "collision_details_log.txt"
    summary_log_path = "summary_log.txt"

    try:
        collision_log = open(collision_log_path, "w")
        summary_log = open(summary_log_path, "w")

        # Write headers
        collision_log.write("Iteration,Collision Indexes\n")
        summary_log.write(
            "Total Iterations,Collisions Detected,No Collisions\n")

        for iteration in range(1, total_iterations + 1):
            print(f"\n=== Iteration {iteration}/{total_iterations} ===")

            if not run_wrapper(orca_dir):
                print(
                    f"[ERROR] Skipping iteration {iteration} due to wrapper.py failure.")
                continue

            print("[INFO] Waiting for 5 seconds...")
            time.sleep(5)

            # Only record first iteration
            record_this_iteration = record_first_iteration and iteration == 1
            if record_this_iteration:
                print("[INFO] Recording video for first iteration...")

            proc_visual = run_visualize(
                visualize_dir, record_video=record_this_iteration)
            collisions, collision_indexes = check_collisions(
                proc_visual, timeout=10)
            terminate_process(proc_visual)

            if collisions:
                print(f"[RESULT] Iteration {iteration}: Collision detected.")
                collision_count += 1
                for idx_set in collision_indexes:
                    collision_log.write(f"{iteration},{{{idx_set}}}\n")
            else:
                print(
                    f"[RESULT] Iteration {iteration}: No collision detected.")

            # Optional: Short pause between iterations
            time.sleep(1)

        # Summary of results
        no_collision_count = total_iterations - collision_count
        print("\n=== Summary ===")
        print(f"Total Iterations: {total_iterations}")
        print(f"Collisions Detected: {collision_count}")
        print(f"No Collisions: {no_collision_count}")

        # Write summary to log
        summary_log.write(
            f"{total_iterations},{collision_count},{no_collision_count}\n")

        # Notify completion
        print("\nAll iterations completed.")
        print(
            f"Detailed collision information is logged in '{collision_log_path}'.")
        print(f"Summary of results is logged in '{summary_log_path}'.")

    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Ensure log files are closed
        if 'collision_log' in locals():
            collision_log.close()
        if 'summary_log' in locals():
            summary_log.close()


if __name__ == "__main__":
    main()
