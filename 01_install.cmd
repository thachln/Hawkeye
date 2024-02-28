@ECHO OFF
rem Create virtual environment
python -m venv venv
rem Active the created virtual environment
call venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install pandas numpy tqdm yacs scikit-image tensorboardX matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

