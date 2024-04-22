# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Main script for hyperparameter optimisation

# Modifiable experiment options.
# Expt options include {volatility, electricity, traffic, favorita}

EXPT1=traffic
EXPT2=electricity

mkdir tft_outputs

OUTPUT_FOLDER=./tft_outputs # Path to store data & experiment outputs
USE_GPU=yes
TESTING_MODE=yes # If yes, trains a small model with little data to test script

# Step 1: Setup environment.
echo
echo Setting up virtual environment...
echo

# set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get install python3-pip || sudo apt-get install python-pip
elif [[ "$OSTYPE" == "msys" ]]; then
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
fi

pip3 install virtualenv || pip install virtualenv # Assumes pip3 is installed!

python -m virtualenv venv || py -m virtualenv venv

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate

elif [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate

else
    echo "Unsupported operating system"
    exit 1
fi

pip3 install -r requirements.txt

# # Step 2: Downloads data if not present.
echo
python -m script_download_data $EXPT1 $OUTPUT_FOLDER
python -m script_download_data $EXPT2 $OUTPUT_FOLDER

# # Step 3: Train & Test
echo
python -m script_train_fixed_params $EXPT1 $OUTPUT_FOLDER $USE_GPU $TEST_MODE
python -m script_train_fixed_params $EXPT2 $OUTPUT_FOLDER $USE_GPU $TEST_MODE

# # Uncomment below for full hyperparamter optimisation.
python3 -m script_hyperparam_opt $EXPT1 $OUTPUT_FOLDER $USE_GPU yes
python3 -m script_hyperparam_opt $EXPT2 $OUTPUT_FOLDER $USE_GPU yes
