# KẾT HỢP SONG TUYẾN CNN-TRANSFORMER CHO PHÂN ĐOẠN ẢNH VẾT NỨT

[![Static Badge](https://img.shields.io/badge/CrackVarious-Dataset?label=Dataset&link=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F16fOIml_hTxCWjRdqZIZGO7Ci9RWjlcJo)](https://drive.google.com/file/d/16fOIml_hTxCWjRdqZIZGO7Ci9RWjlcJo)

Kho lưu trữ này chứa mã nguồn PyTorch của CrackAwareFusionNet (CAFNet) cho bài toán phân đoạn vết nứt ở mức độ điểm ảnh (pixel-wise) trên các hình ảnh hạ tầng dân dụng.

## Cập nhật
**2026-01-01**
- 
## 1. Kiến trúc mô hình

![](/figures/thesis_architecture.jpg)

- Mô hình đã huấn luyện (Trained Model): [Trọng số (Weight)](https://drive.google.com/drive/folders/1q4I_aAuuMAhBIGqxVPUmcEtdkdHnnISS?usp=sharing)

## 2. Bộ dữ liệu
Bộ dữ liệu CrackVarious được tổ chức theo cấu trúc sau:

<CRACKVARIOUS_ROOT>/
  train/
    IMG/
    GT/
  val/
    IMG/
    GT/
  test/
    IMG/
    GT/

**Liên kết tải xuống:**
- Toàn bộ dữ liệu: [CrackVarious](https://drive.google.com/file/d/16fOIml_hTxCWjRdqZIZGO7Ci9RWjlcJo/view?usp=sharing)
- Các tập dữ liệu thành phần: [Pavement](https://drive.google.com/file/d/12n5K7Fcb74vT589RJ1d6uVf8JrwX1A7t/view?usp=sharing) | [Masonry](https://drive.google.com/file/d/1iDTkXXFvd1RGcljVU3JDVLILBnkGkVya/view?usp=sharing) | [Steel](https://drive.google.com/file/d/1Eq95Cr54m7PPtrKWNbS3b8aLZ6CbG9FU/view?usp=sharing)

Các độ đo (metrics) và mã nguồn đánh giá được đặt trong file `CrackAwareFusionNet/metric.py`.

## 3. Kết quả

### Kết quả trên tập dữ liệu CrackVarious

| Mô hình (Model) | Precision (%) | Recall (%) | F1-Score (%) | mIoU (%) |
| :--- | :---: | :---: | :---: | :---: |
| U-Net | 78.88 | 83.34 | 80.86 | 68.11 |
| SegNet | 80.31 | 82.90 | 81.40 | 68.88 |
| DeepLabV3+ | 80.86 | 81.07 | 80.78 | 67.99 |
| SegFormer-B1 | 79.81 | 82.94 | 81.12 | 68.51 |
| HrSegNet | 77.29 | 72.92 | 74.76 | 59.94 |
| HACNetV2 | 81.00 | 80.36 | 80.46 | 67.55 |
| DTrc-Net | 82.59 | 80.79 | 81.48 | 68.99 |
| HybridSegmentor | 82.85 | 77.92 | 80.31 | 67.10 |
| **CAFNet (Ours)** | **81.54** | **84.26** | **82.72** | **70.71** |

### So sánh với các mô hình State-of-the-art
![](/figures/compare_sota.jpg)
