#!bin/bash
poetry shell
poetry install
cd third_party/ctcdecode && python setup.py install
python setup.py install # i dont know why ctcdecode cant install from the first time
cd ../..
pre-commit install
pre-commit autoupdate
git switch develop
