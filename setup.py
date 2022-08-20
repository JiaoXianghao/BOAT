from setuptools import setup,find_packages
from os import path

current_directory = path.abspath(path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_install_requirements():
    requirements_path = path.join(current_directory, "requirements.txt")
    with open(requirements_path, encoding="utf-8") as fp:
        return fp.read().splitlines()


setup(
    name="bohml",
    version="0.1.1",
    packages=find_packages(),
    long_description=long_description,
    url="https://github.com/JiaoXianghao/BOHML",
    license="MIT",
    keywords=[
    "bilevel-optimization",
    "lerning and vision",
    "python",
    "Deep learning"],
    install_requires=get_install_requirements(),
    classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    ],
    python_requires='>=3.6.0',
    author="Xianghao Jiao, Yaohua Liu, Risheng Liu",
    author_email="jxh@mail.dlut.edu.cn",
    description="A Bilevel Optimization Toolkit in Python for Learning and Vision Tasks",
)


