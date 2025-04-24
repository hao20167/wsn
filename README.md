# Wireless Sensor Network (WSN) Localization

## Giới thiệu
Dự án này nhằm giải quyết bài toán định vị các nút chưa biết (unknown nodes) trong mạng cảm biến không dây (WSN - Wireless Sensor Network) bằng các thuật toán khác nhau.

## Cấu trúc thư mục
```
.
├── input           # Thư mục chứa dữ liệu đầu vào (được sinh bởi chương trình)
├── output          # Thư mục chứa kết quả đầu ra (tọa độ các unknown nodes)
├── main.py         # File thực thi chính
├── README.md       # Tài liệu hướng dẫn
└── ...             # Các file source code liên quan
```

## Các thuật toán đã sử dụng
1. **DV-Hop**: Chỉ dùng thuật toán DV-Hop cải tiến.
2. **VVS-HCO**: Thuật toán kết hợp Virtual Viscosity Strategy với Human Conception Optimization.
3. **HCO-DPSO**: Kết hợp thuật toán Human Conception Optimization với Discrete Particle Swarm Optimization.
4. **VVS-HCO Custom**: Phiên bản cải tiến của VVS-HCO, thay vì khởi tạo quần thể sperm hoàn toàn ngẫu nhiên, sẽ lấy 50% kết quả từ DV-Hop và 50% ngẫu nhiên.

## Thông số chạy thử nghiệm
- **Số test**: 10
- **Số lượng node**: 100
- **Tỷ lệ anchor node**: 10%
- **Phạm vi giao tiếp (communication range)**: 15
- **Diện tích vùng triển khai**: 100x100

## Kết quả mẫu

| Input   | DV-Hop | VVS-HCO | HCO-DPSO | VVS-HCO Custom |
|---------|--------|---------|----------|----------------|
| 1.txt   | 10.005 | 8.029   | 7.472    | 8.130          |
| 2.txt   | 24.419 | 6.896   | 6.526    | 7.100          |
| 3.txt   | 14.396 | 36.624  | 36.594   | 36.530         |
| 4.txt   | 66.355 | 35.400  | 34.885   | 35.188         |
| 5.txt   | 24.563 | 36.382  | 35.916   | 36.530         |
| 6.txt   | 13.683 | 8.665   | 8.023    | 8.580          |
| 7.txt   | 19.395 | 7.546   | 7.004    | 7.466          |
| 8.txt   | 24.766 | 15.442  | 15.050   | 15.481         |
| 9.txt   | 27.601 | 5.874   | 5.965    | 6.287          |
| 10.txt  | 15.912 | 34.812  | 34.340   | 34.774         |

## Hướng dẫn chạy chương trình
Nếu bạn muốn chạy lại thử nghiệm:

1. Clone hoặc tải xuống repository này.
2. Xóa hai thư mục `input` và `output` hiện có.
3. Chỉnh sửa các thông số đầu vào trong file `main.py` theo nhu cầu của bạn.
4. Chạy chương trình bằng cách:
```bash
python main.py
```

Chương trình sẽ tự động tạo dữ liệu đầu vào mới và tính toán kết quả, lưu vào thư mục `output`.