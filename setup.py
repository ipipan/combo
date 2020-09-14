"""Setup."""
from setuptools import find_packages, setup

REQUIREMENTS = [
    'absl-py==0.9.0',
    'allennlp==1.1.0',
    'conllu==2.3.2',
    'dataclasses-json==0.5.2',
    'joblib==0.14.1',
    'jsonnet==0.15.0',
    'requests==2.23.0',
    'overrides==3.1.0',
    'tensorboard==2.1.0',
    'torch==1.6.0',
    'tqdm==4.43.0',
    'transformers>=3.0.0,<3.1.0',
    'urllib3==1.24.2',
]

setup(
    name='COMBO',
    version='0.0.1',
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=['tests']),
    setup_requires=['pytest-runner', 'pytest-pylint'],
    tests_require=['pytest', 'pylint'],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['combo = combo.main:main']},
)
