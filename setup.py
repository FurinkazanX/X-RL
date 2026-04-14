"""项目安装脚本"""

from setuptools import setup, find_packages


setup(
    name="xrl",
    version="0.1.0",
    description="Distributed Reinforcement Learning Framework based on Ray",
    author="X-RL Team",
    author_email="xrl@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ray[default]>=2.9.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "gymnasium>=0.29.0",
        "tensorboard>=2.14.0",
        "wandb>=0.15.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "xrl-train = xrl.main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8"
)
