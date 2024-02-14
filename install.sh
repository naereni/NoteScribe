#!bin/bash
env_path="$(poetry env use python3.9)"
source ${env_path:18}/bin/activate
poetry install
cd third_party/ctcdecode && python setup.py install
python setup.py install 
cd ../..
pre-commit install
pre-commit autoupdate
git switch develop
