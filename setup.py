from setuptools import find_packages, setup

__version__ = "3.0.0b15"

with open("README.md", "r") as fh:

    long_description = fh.read()

setup(
    name="humanoidgen",
    version=__version__,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zhi Jing, Siyuan Yang, Jicong Ao, Ting Xiao, Yu-Gang Jiang, Chenjia Bai",
    url="https://github.com/TeleHuman/HumanoidGen",
    packages=find_packages(include=["humanoidgen*"]),
    python_requires=">=3.9",
    setup_requires=["setuptools>=62.3.0"],
    package_data={},
)