# Hệ thống Phát hiện và Phân Đoạn Vết Nứt (Crack Detection and Segmentation)

## 📋 Mô tả Dự án

Đây là một dự án học sâu (Deep Learning) được xây dựng để phát hiện và phân đoạn vết nứt trên bề mặt các vật liệu xây dựng. Dự án sử dụng các kiến trúc mạng nơ-ron hiện đại như CAFNet, CrackAwareFusionNet, ResNet50 và MobileNetV3 kết hợp với PyTorch Lightning để huấn luyện và đánh giá mô hình.

### Các tính năng chính:
- 🏗️ Hỗ trợ nhiều kiến trúc mô hình (CAFNet, ResNet50, MobileNetV3, v.v.)
- 📊 Hệ thống cấu hình linh hoạt sử dụng YAML
- 🎯 Hỗ trợ batch processing và inference
- 📈 Logging chi tiết với TensorBoard
- ✅ Đánh giá toàn diện với các metric: Accuracy, F1-Score, Precision, Recall, IoU, Dice Loss
- 💾 Checkpointing tự động và Early Stopping

---

## 📁 Cấu trúc Thư mục

```
.
├── config/                      # Cấu hình dự án
│   ├── model.yaml              # Cấu hình các mô hình
│   └── train.yaml              # Cấu hình huấn luyện
├── data/                        # Xử lý dữ liệu
│   ├── __init__.py
│   ├── dataset.py              # Class Dataset tùy chỉnh
│   └── pldatamodule.py         # PyTorch Lightning DataModule
├── models/                      # Các kiến trúc mô hình
│   ├── __init__.py
│   ├── CAFNet_mbnv3l.py        # CAFNet với MobileNetV3-Large
│   ├── CrackModule.py          # Lightning Module chính
│   ├── MiT.py                  # Mix Transformer Backbone
│   ├── ResNet.py               # ResNet Backbone
│   ├── mobilenetv3.py          # MobileNetV3 Backbone
│   ├── decoder.py              # Decoder cho phân đoạn
│   ├── proposed.py             # Mô hình đề xuất
│   ├── registry.py             # Registry để quản lý mô hình
│   └── __pycache__/
├── utils/                       # Các tiện ích
│   ├── __init__.py
│   └── metric.py               # Hàm loss và metrics tùy chỉnh
├── tests/                       # Jupyter notebooks demo
│   ├── demo.ipynb
│   └── terminal.ipynb
├── checkpoints/                 # Lưu trữ weight mô hình
├── logs/                        # TensorBoard logs
├── train.py                     # Script huấn luyện
├── evaluate.py                  # Script đánh giá
├── predict.py                   # Script inference/dự đoán
├── requirements.txt             # Các thư viện cần thiết
└── README.md                    # File này
```

---

## 🚀 Cài đặt

### 1. Yêu cầu hệ thống
- Python >= 3.8
- CUDA 11.0+ (nếu sử dụng GPU)

### 2. Cài đặt các thư viện

```bash
# Tạo virtual environment (tùy chọn nhưng khuyến nghị)
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

### 3. Các thư viện chính
- **PyTorch**: `torch>=1.10.0` - Framework học sâu
- **PyTorch Lightning**: `pytorch-lightning>=2.6.0` - Huấn luyện dễ dàng
- **OpenCV**: `opencv-python>=4.5.0` - Xử lý ảnh
- **NumPy**: `numpy>=1.20.0` - Tính toán khoa học
- **OmegaConf**: `omegaconf` - Quản lý cấu hình
- **torchvision**: `torchvision>=0.11.0` - Các mô hình pretrained

---

## 📖 Hướng dẫn Sử dụng

### 1. Huấn luyện Mô hình

Sửa file cấu hình `config/train.yaml` và `config/model.yaml` theo nhu cầu của bạn, sau đó chạy:

```bash
python train.py
```

**Các tùy chọn cấu hình:**
- `config/model.yaml`: Chọn mô hình, số lớp, tham số học
- `config/train.yaml`: Số epoch, batch size, learning rate, data path

### 2. Đánh giá Mô hình

```bash
python evaluation.py
```

Các metrics được tính:
- **Accuracy**: Độ chính xác tổng thể
- **F1-Score**: Trung bình điều hòa của Precision và Recall
- **Precision**: Độ chính xác dương tính
- **Recall**: Khả năng phát hiện
- **IoU (Jaccard Index)**: Giao trên hợp
- **Dice Loss**: Loss function chủ yếu

### 3. Dự đoán trên ảnh mới

```bash
python predict.py --image_path <đường_dẫn_ảnh> --checkpoint <đường_dẫn_model>
```

---

## 🏗️ Các Mô hình Được Hỗ trợ

| Tên Mô hình | Mô tả | Đặc điểm |
|------------|-------|---------|
| `CAFNet` | Crack-Aware Fusion Network | Kết hợp đặc trưng đa tầng |
| `CAFNet_MBNV3L` | CAFNet với MobileNetV3-Large | Nhẹ nhàng, nhanh chóng |
| `ResNet50` | Residual Network 50 layers | Mạnh mẽ, độ chính xác cao |
| `MobileNetV3` | MobileNetV3 | Hiệu quả trên thiết bị di động |
| `MiT` | Mix Transformer | Transformer-based backbone |

---

## 📊 Giám sát Huấn luyện

TensorBoard logs được lưu trong thư mục `logs/`. Để xem logs:

```bash
tensorboard --logdir=logs/
```

Sau đó truy cập `http://localhost:6006` trên trình duyệt.

---

## 📦 Kết Quả Huấn luyện

Các checkpoint model tốt nhất sẽ được lưu trong thư mục `checkpoints/`. Các file log chi tiết được lưu trong `logs/`.

---

## 🔧 Cấu hình

### `config/model.yaml`
Định nghĩa các tham số mô hình:
```yaml
cafnet:
  name: "CAFNet_MBNV3L"
  hparams:
    num_classes: 1
    encoder_channels: [...]
```

### `config/train.yaml`
Định nghĩa các tham số huấn luyện:
```yaml
trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: [0]

optim:
  lr: 0.0001
  weight_decay: 0.00001
```

---

## 🤝 Đóng góp

Nếu bạn muốn cải thiện dự án này:

1. Fork repository
2. Tạo branch cho feature của bạn (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📝 Ghi chú

- Đảm bảo dữ liệu training được đặt trong thư mục đúng như cấu hình
- Kiểm tra GPU availability trước huấn luyện: `python -c "import torch; print(torch.cuda.is_available())"`
- Backup checkpoint thường xuyên trong quá trình huấn luyện

---

## 📞 Liên hệ

Nếu có câu hỏi hoặc vấn đề gì, vui lòng tạo một issue trong repository.

---

## 📄 License

Dự án này được cấp phép dưới [Chondra lại thêm license của bạn ở đây]

---

**Cập nhật lần cuối**: Tháng 3, 2026
