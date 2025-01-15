import os
import numpy as np
import matlab.engine
from example_orca import example_orca
from utils import parse_config


def search_grad_python(
    pool_f: str,
    val: float,
    max_ite: int,
    f_out: str,
    dur: float,
    seed: int,
    matlab_root: str,
    step_size: float = 0.01
) -> None:
    """
    Gradient descent search for optimal attack parameters.

    Args:
        pool_f: Path to seedpool file
        val: GPS spoofing deviation
        max_ite: Maximum iterations
        f_out: Output results file
        dur: Initial attack duration
        seed: Random seed
        matlab_root: Path to MATLAB root directory
        step_size: Step size for gradient calculation
    """
    # Setup MATLAB engine
    eng = matlab.engine.start_matlab()
    fuzz_dir = os.path.join(matlab_root, "fuzz")
    eng.addpath(fuzz_dir)

    try:
        # Setup file paths
        tmp_files = {
            'pos': os.path.join(fuzz_dir, f'tmp_files/attack_pos{seed}.csv'),
            'dist': os.path.join(fuzz_dir, f'tmp_files/dist_obs{seed}.csv'),
            'col': os.path.join(fuzz_dir, f'tmp_files/nb_col{seed}.csv'),
            'info': os.path.join(fuzz_dir, f'tmp_files/col_info{seed}.csv'),
            'parent': os.path.join(fuzz_dir, f'tmp_files/parent{seed}.csv'),
            'neigh': os.path.join(fuzz_dir, f'tmp_files/neighbor{seed}.csv')
        }

        # Read seedpool
        seedpool = np.loadtxt(pool_f, delimiter=',')
        if seedpool.ndim == 1:  # If only one row
            seedpool = seedpool.reshape(1, -1)

        # Results matrix: [final_col, init_fitness, fitness, attack_start, duration, attack_id, victim_id, deviation]
        out_mat = np.zeros((len(seedpool), 8))

        # Read configuration for end time
        # TODO: change the path to the config file
        config = parse_config(
            "../False_Data_Attacks_on_ORCA/Attack/src/config.txt")
        time_step = float(config.get("time_step", 0.1))
        end_time = float(config.get("totalTimeStep", 200)) * time_step

        for row, seed_params in enumerate(seedpool):
            att_id = int(seed_params[0])
            vim_id = int(seed_params[1])
            dev_y = val * seed_params[2]
            st_init = seed_params[3]
            t_init = dur
            ite = 0

            print(f"\nProcessing seed {seed}, row {row+1}/{len(seedpool)}")
            print(f"Initial params: start={st_init:.3f}, dur={t_init:.3f}")

            while ite < max_ite:
                # 1. Evaluate current point
                example_orca(st_init, t_init, att_id, vim_id, dev_y, seed,
                             tmp_files['pos'], tmp_files['dist'],
                             tmp_files['col'], tmp_files['info'])

                # Use MATLAB's cal_summary
                eng.cal_summary(tmp_files['col'],
                                tmp_files['info'],
                                tmp_files['parent'],
                                nargout=0)

                # Read results
                record = np.loadtxt(tmp_files['parent'], delimiter=',')
                nb_col = int(record[2])
                fitness = record[4]

                if ite == 0:
                    init_fitness = fitness

                # Check for collision
                if nb_col == 1:
                    print(f"Collision achieved! Time: {record[3]:.3f}")
                    out_mat[row] = [1, init_fitness, fitness, record[0],
                                    record[1], att_id, vim_id, dev_y]
                    break

                # 2. Calculate gradients using finite differences
                # Gradient for start time
                example_orca(st_init + step_size, t_init, att_id, vim_id, dev_y, seed,
                             tmp_files['pos'], tmp_files['dist'],
                             tmp_files['col'], tmp_files['info'])
                eng.cal_summary(tmp_files['col'],
                                tmp_files['info'],
                                tmp_files['neigh'],
                                nargout=0)
                record_st = np.loadtxt(tmp_files['neigh'], delimiter=',')

                # Gradient for duration
                example_orca(st_init, t_init + step_size, att_id, vim_id, dev_y, seed,
                             tmp_files['pos'], tmp_files['dist'],
                             tmp_files['col'], tmp_files['info'])
                eng.cal_summary(tmp_files['col'],
                                tmp_files['info'],
                                tmp_files['neigh'],
                                nargout=0)
                record_t = np.loadtxt(tmp_files['neigh'], delimiter=',')

                # 3. Calculate gradients (match MATLAB exactly)
                dst = (record_st[4] - fitness) / step_size
                dt = (record_t[4] - fitness) / step_size

                # Avoid division by zero (same as MATLAB)
                dst = 1e-6 if abs(dst) < 1e-6 else dst
                dt = 1e-6 if abs(dt) < 1e-6 else dt

                # 4. Update parameters (match MATLAB exactly)
                loss = fitness - 0.5
                k = max(abs(dst), abs(dt))
                lr = loss / (2 * k * k)

                st_new = st_init - lr * dst
                t_new = t_init - lr * dt

                print(f"Iteration {ite+1}: loss={loss:.3f}, lr={lr:.3f}")
                print(f"Updated: start={st_new:.3f}, dur={t_new:.3f}")

                # Check validity (same as MATLAB)
                if (st_new + t_new > end_time) or (st_new < 0):
                    print(
                        f"Parameters out of valid range: start={st_new:.3f}, dur={t_new:.3f}")
                    print(f"Valid range: 0 ≤ start + duration ≤ {end_time}")
                    break

                st_init = st_new
                t_init = t_new
                ite += 1

            # Record results if no collision found
            if nb_col != 1:
                out_mat[row] = [0, init_fitness, fitness, st_init,
                                t_init, att_id, vim_id, dev_y]

        # Save non-zero results
        valid_results = out_mat[np.any(out_mat != 0, axis=1)]
        if len(valid_results) > 0:
            np.savetxt(f_out, valid_results, delimiter=',')
            print(f"\nResults saved to {f_out}")
        else:
            print("\nNo valid results to save")

    finally:
        eng.quit()


if __name__ == "__main__":
    # Example usage
    # TODO: change the path to the config file
    MATLAB_ROOT = "../comparison/swarmlab"
    search_grad_python(
        pool_f="path/to/seedpool.csv",
        val=5.0,
        max_ite=2,
        f_out="path/to/results.csv",
        dur=0.1,
        seed=200,
        matlab_root=MATLAB_ROOT
    )
