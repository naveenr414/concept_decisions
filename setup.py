from setuptools import find_packages, setup

setup(
    name='concept_abstraction',
    version='1.0.0',
    description='Decision-Relevant Concept Selection for Reinforcement Learning',
    author='Naveen Raman, Stephanie Milani, Fei Fang',
    python_requires='>=3.10',
    packages=find_packages(include=['concept_abstraction']),
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'stable-baselines3',
        'gymnasium',
        'gurobipy',
        'scikit-learn',
        'ujson',
        'opencv-python-headless',
        'matplotlib',
        'seaborn',
    ],
)