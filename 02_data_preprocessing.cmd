rem kích hoạt biến môi trường
call .env\Scripts\activate
rem Giải nén file data/dataset.zip chạy trên windows sau đó copy dataset giải nen vào thư mục data
pip install scikit-image
rem chạy lệnh sau để tăng cường dữ liệu
python dataset/data_augmentation.py

