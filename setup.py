from setuptools import setup

extras_require = [
    'ray[rllib,debug]==0.7.3',
    'tensorflow==1.13.0rc1',
    'pillow==6.1.0',
    'pandas==0.25.0',
    'requests==2.22.0',
    'configspace==0.4.10',
    'scipy==1.3.0',
    'pandas==0.25.0',
]

extras_require_cont = [
    # 'ray[rllib,debug]==0.9.0',
    'tensorflow==2.2.0',
    'tensorflow-probability==0.9.0',
    'pandas==0.25.0',
    'requests==2.22.0',
    'mujoco-py==2.0.2.13', # with mujoco 2.0
    'configspace==0.4.10',
    'scipy==1.3.0',
    'pandas==0.25.0',
]

setup(name='mdp_playground',
      version='0.0.1',
      install_requires=['gym'],
      extras_require={
          'extras_disc': extras_require,
          'extras_cont': extras_require_cont,
      },
)
