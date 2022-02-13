#!/bin/bash

INPUT_PATH="input/feedback-prize-2021"

# create ./input/data directory (recursively) if it doesn't exist
if [ ! -d $INPUT_PATH ]; then
    echo "Creating $INPUT_PATH directory"
    mkdir -p $INPUT_PATH 
fi


cd $INPUT_PATH
kaggle competitions download -c feedback-prize-2021
unzip feedback-prize-2021.zip
rm feedback-prize-2021.zip
cd -
