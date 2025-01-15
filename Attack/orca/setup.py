from setuptools import setup, Extension
import pybind11
from distutils import sysconfig

eigen_include_dir = "/usr/local/include/eigen3"
rvo_include_dir = "../RVO2/src"
other_include_dir = "../Attack/include"

cfg_vars = sysconfig.get_config_vars()
cfg_vars['CFLAGS'] = cfg_vars['CFLAGS'].replace('-arch arm64', '')
cfg_vars['LDFLAGS'] = cfg_vars['LDFLAGS'].replace('-arch arm64', '')

setup(
    name='orca_module',
    ext_modules=[
        Extension(
            'orca_module',
            ['wrapper.cpp'],
            include_dirs=[
                pybind11.get_include(),
                eigen_include_dir,
                rvo_include_dir,
                other_include_dir
            ],
            library_dirs=[
                '../RVO2/build/src',
                '../Attack/build'
            ],
            libraries=[
                'RVO',
                'attacklib_RunningAlgorithm',
                'attacklib_ORCARunningAlgorithm',
                # any other required libraries
            ],
            extra_compile_args=["-std=c++11"],
            extra_link_args=["-arch", "x86_64"]
        ),
    ],
)
