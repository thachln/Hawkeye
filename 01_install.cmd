rem Tạo biến môi trường
python -m venv .env
rem kích hoạt biến môi trường
call .env\Scripts\activate
rem Requirements
pip install numpy
pip install tqdm
pip install yac
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

