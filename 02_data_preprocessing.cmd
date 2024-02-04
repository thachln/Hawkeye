@ECHO OFF
rem Prepare data/dataset by uncompress file data/dataset.zip into data/dataset

rem Activate the virtual environment
call venv\Scripts\activate
rem update pip
python -m pip install --upgrade pip

rem Perform data augmentation
python dataset/data_augmentation.py
rem Generate metadata: train.txt, val.txt 
python dataset/metadata.py

