import os
from setuptools import setup
from setuptools import find_packages

setup(
    name="cablab",
    version="0.9",
    description="Cablab",
    install_requires=[],
    packages=find_packages(),
    package_dir={"cablab": "cablab"},
    package_data={
        "": ["*.toml"],
        "/cablab/configs": ["/cablab/configs/*.so"],
    },
    data_files=[
        ("/cablab/configs", ["/cablab/configs/dqn_conf.toml"]),
        ("/cablab/configs", ["/cablab/configs/ma_dqn_conf.toml"]),
        ("/cablab/configs", ["/cablab/configs/ppo_conf.toml"]),
    ]
)
