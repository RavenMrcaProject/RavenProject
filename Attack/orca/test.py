import matlab.engine
import os


def test_matlab_integration():
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()

    # Get current directory and add paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fuzz_dir = os.path.join(root_dir, 'fuzz')

    print(f"Adding paths:\n{fuzz_dir}")
    eng.addpath(fuzz_dir)
    eng.addpath(os.path.join(fuzz_dir, 'prepare'))
    eng.addpath(os.path.join(fuzz_dir, 'search'))
    eng.addpath(os.path.join(fuzz_dir, 'seedpools'))

    # Test MATLAB function call
    print("\nTesting MATLAB calculation...")
    result = eng.eval('2+2')
    print(f"2 + 2 = {result}")

    # Test if we can access the fuzz directory
    print("\nTesting MATLAB file access...")
    # Use ls instead of dir and convert to string
    eng.eval(f"cd('{fuzz_dir}')")
    files = eng.eval("string(ls)")
    print(f"Files in fuzz directory:\n{files}")

    # Test if we can call prepare.m
    print("\nTesting if prepare.m is accessible...")
    if eng.exist('prepare.m', 'file') == 2:  # 2 means M-file
        print("prepare.m found!")
    else:
        print("prepare.m not found in MATLAB path")

    print("\nClosing MATLAB engine...")
    eng.quit()


if __name__ == "__main__":
    test_matlab_integration()
