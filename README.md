# custom-gym-env

conda env create -n <env_name> -f py36_toy_rl.yml
git clone git@github.com:RaghuSpaceRajan/custom-gym-env.git
cd custom-gym-env
pip install -e .
git clone git@github.com:RaghuSpaceRajan/gym-extension.git
cd gym-extension
pip install -e .

It is recommended to OHE the states/actions before using them in a function approximator.
