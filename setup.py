"""Setup."""
from setuptools import find_packages, setup

REQUIREMENTS = [
    'absl-py==0.9.0',
    'allennlp==1.0.0rc5',
    'conllu==2.3.2',
    'joblib==0.14.1',
    'jsonnet==0.15.0',
    'overrides==3.0.0',
    'tensorboard==2.1.0',
    'torch==1.5.0',
    'torchvision==0.6.0',
    'transformers==2.9.1',
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
