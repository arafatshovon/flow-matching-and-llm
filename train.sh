#!/bin/bash

echo "$(pwd)"
echo -e "activating env"
source flow_env/bin/activate

jupyter nbconvert --to notebook --execute notebooks/Flow_Matching.ipynb --ExecutePreprocessor.timeout=-1 --output Flow_Matching_output.ipynb 2>&1 | tee execution.log
# rm -rf notebooks/lightning_logs

deactivate


