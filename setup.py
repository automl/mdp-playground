from setuptools import setup

extras_require = [
    'ray[rllib,debug]==0.7.3',
    'tensorflow==1.13.0rc1',
    'pandas==0.25.0',
    'requests==2.22.0',
]

setup(name='mdp_playground',
      version='0.0.1',
      install_requires=['gym_extensions_for_mdp_playground'],
      extras_require={
          'extras': extras_require,
      },
)
