import torch
from torchvision import transforms, datasets

# Đường dẫn đến thư mục chứa dữ liệu của bạn
data_dir = 'data/dataset'

# Định nghĩa transform để chuyển đổi ảnh thành dạng tensor và chuẩn hóa
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# Tạo dataset sử dụng ImageFolder
custom_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Tính toán mean và variance
loader = torch.utils.data.DataLoader(custom_dataset, batch_size=len(custom_dataset), shuffle=False)
data = next(iter(loader))
mean_values = data[0].mean(dim=(0, 2, 3))
variance_values = data[0].var(dim=(0, 2, 3))

# Hiển thị giá trị mean và variance
print("Mean values:", mean_values)
print("Variance values:", variance_values)

# Áp dụng transform Normalize với dữ liệu của bạn
custom_transform = transforms.Compose([
    transforms.Normalize(mean=mean_values.tolist(), std=variance_values.sqrt().tolist())
])

# Áp dụng transform cho dữ liệu của bạn
normalized_custom_data = custom_transform(data[0])

# In giá trị mean và variance sau khi áp dụng Normalize
print("Mean values after normalization:", normalized_custom_data.mean(dim=(0, 2, 3)))
print("Variance values after normalization:", normalized_custom_data.var(dim=(0, 2, 3)))
