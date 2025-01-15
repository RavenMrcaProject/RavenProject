import os
from typing import Tuple, Dict
import numpy as np
import matlab.engine
from example_orca import example_orca, read_obstacles, get_obstacle_center
from search_grad_python import search_grad_python


def setup_matlab_paths() -> Dict[str, str]:
    """Setup and return MATLAB directory paths"""
    # TODO: change the path to the config file
    MATLAB_ROOT = "../comparison/swarmlab"
    FUZZ_DIR = os.path.join(MATLAB_ROOT, "fuzz")

    paths = {
        'root': MATLAB_ROOT,
        'fuzz': FUZZ_DIR,
        'tmp_files': os.path.join(FUZZ_DIR, 'tmp_files'),
        'prepare': os.path.join(FUZZ_DIR, 'prepare'),
        'search': os.path.join(FUZZ_DIR, 'search'),
        'seedpools': os.path.join(FUZZ_DIR, 'seedpools'),
        'attResults': os.path.join(FUZZ_DIR, 'attResults')
    }
    return paths


def get_file_paths(paths: Dict[str, str], seed: int) -> Dict[str, str]:
    """Generate file paths for a given seed"""
    tmp_files = {
        'pos': os.path.join(paths['tmp_files'], f'attack_pos{seed}.csv'),
        'dist': os.path.join(paths['tmp_files'], f'dist_obs{seed}.csv'),
        'col': os.path.join(paths['tmp_files'], f'nb_col{seed}.csv'),
        'info': os.path.join(paths['tmp_files'], f'col_info{seed}.csv'),
        'parent': os.path.join(paths['tmp_files'], f'parent{seed}.csv'),
        'neigh': os.path.join(paths['tmp_files'], f'neighbor{seed}.csv')
    }

    output_files = {
        'ite': os.path.join(paths['search'], f'iteration{seed}.csv'),
        'cond': os.path.join(paths['search'], f'condition{seed}.csv'),
        'dire': os.path.join(paths['prepare'], f'dire{seed}.csv'),
        'seedpool': os.path.join(paths['seedpools'], f'pool{seed}.csv'),
        'results': os.path.join(paths['attResults'], f'att_results{seed}.csv')
    }

    return {**tmp_files, **output_files}


def cleanup_files(files: Dict[str, str]) -> None:
    """Clean up temporary files"""
    for file_path in files.values():
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {str(e)}")


def swarmfuzz(
    seed_start: int,
    seed_end: int,
    dev: float,
    nb: int,
    max_iterations: int = 50,
    attack_duration: float = 2.0,
    step_size: float = 0.01  # For gradient descent
) -> None:
    """
    Main SwarmFuzz function that coordinates the fuzzing process.

    Args:
        seed_start: Starting seed number
        seed_end: Ending seed number
        dev: Deviation parameter
        nb: Number of robots
        max_iterations: Maximum gradient descent iterations
        attack_duration: Initial attack duration
        step_size: Step size for gradient calculation
    """
    # Setup paths and constants
    paths = setup_matlab_paths()

    # Initialize MATLAB engine
    try:
        eng = matlab.engine.start_matlab()
        eng.addpath(paths['fuzz'])
        print("MATLAB engine initialized successfully")

        for seed in range(seed_start, seed_end + 1):
            print(f"\nProcessing seed {seed}")

            # Get file paths for this seed
            files = get_file_paths(paths, seed)

            # Clean up any existing files
            cleanup_files(files)

            try:
                # 1. Run ORCA simulation (no attack)
                print(f"Running no-attack simulation for seed {seed}")
                example_orca(0, 0, 0, 0, 0, seed,
                             files['pos'], files['dist'],
                             files['col'], files['info'])

                # Get obstacle center for MATLAB prepare
                # TODO: change the path to the config file
                obstacles = read_obstacles(
                    "../False_Data_Attacks_on_ORCA/Attack/environments/env-orca-1_comparison2/obstacles.txt")
                center = get_obstacle_center(obstacles[0])
                c_obs = matlab.double([[center[0]], [center[1]]])

                # 2. Run MATLAB prepare
                print("Running MATLAB prepare")
                start_t, pos_att, dist_obs, final_dire = eng.prepare(
                    float(nb),
                    c_obs,  # Using center for prepare
                    files['pos'],
                    files['dist'],
                    files['dire'],
                    files['col'],
                    float(seed),
                    nargout=4
                )

                if start_t == -1:
                    print(f"Seed {seed}: Invalid scenario from prepare")
                    continue

                # 3. Run MATLAB seed_gen
                print("Running MATLAB seed_gen")
                flag = eng.seed_gen(
                    float(nb),
                    float(start_t),
                    matlab.double(pos_att),
                    matlab.double(dist_obs),
                    matlab.double(final_dire),
                    files['seedpool'],
                    nargout=1
                )

                if flag <= 0:
                    print(f"Seed {seed}: No valid seedpool generated")
                    continue

                # 4. Run gradient descent search
                if os.path.exists(files['seedpool']):
                    print("*********** Gradient descent ***********")
                    search_grad_python(
                        pool_f=files['seedpool'],
                        val=dev,
                        max_ite=max_iterations,
                        f_out=files['results'],
                        dur=attack_duration,
                        seed=seed,
                        matlab_root=paths['root'],
                        step_size=step_size
                    )

                print(f"Seed {seed}: Successfully processed")

            except Exception as e:
                print(f"Error processing seed {seed}: {str(e)}")
                continue

            finally:
                # Clean up temp files for this seed
                cleanup_files({k: v for k, v in files.items()
                               if k in ['pos', 'dist', 'col', 'info', 'parent', 'neigh']})

    except Exception as e:
        print(f"Fatal error: {str(e)}")

    finally:
        # Clean up MATLAB engine
        if 'eng' in locals():
            eng.quit()
            print("MATLAB engine closed")


if __name__ == "__main__":
    swarmfuzz(
        seed_start=200,
        seed_end=201,
        dev=5.0,
        nb=3,
        max_iterations=100,
        attack_duration=1.0,
        step_size=0.01
    )
