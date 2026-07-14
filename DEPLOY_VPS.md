# Chạy Flower server trên VPS bằng Docker Compose

Server chỉ tổng hợp trọng số trên CPU. Dataset và quá trình huấn luyện vẫn nằm ở từng client.

## 1. Chuẩn bị VPS

Cài Docker Engine và Docker Compose plugin, sau đó chép repository lên VPS. Tại thư mục dự án:

```bash
docker compose build
docker compose up -d
docker compose logs -f fl-server
```

Trước lần chạy đầu tiên, tạo file môi trường và đổi mật khẩu dashboard:

```bash
cp .env.example .env
nano .env
```

Mã nguồn `server/` và `yolov5/` được đóng vào image. Kiểm tra source trước khi build:

```bash
test -f yolov5/models/yolo.py && echo "YOLOv5 source OK"
```

Sau khi sửa Python, dashboard hoặc YOLOv5, cần build lại image. Riêng
`server/config_server.json` được mount từ VPS nên chỉ cần restart container.

Mở `http://IP_VPS:5000`, sau đó đăng nhập bằng `FL_ADMIN_USER` và
`FL_ADMIN_PASSWORD` trong `.env`. Trên giao diện bạn có thể:

- đặt tên test, số round và số client tối thiểu;
- sửa cấu hình epochs/batch/workers cho từng client;
- bắt đầu experiment mới (lượt đang chạy sẽ được dừng trước);
- dừng server, xem log trực tiếp và lịch sử kết quả.

Mặc định server công khai cổng TCP `8080`. Nếu muốn dùng cổng khác ở phía VPS:

```bash
FL_PORT=18080 docker compose up -d
```

Khi đó cần mở TCP `18080` trên firewall/security group và client phải kết nối tới cổng đó.

## 2. Cấu hình phiên huấn luyện

Sửa `server/config_server.json` trước khi khởi động. `min_clients_connected` là số client tối thiểu phải online đồng thời; server sẽ chờ nếu chưa đủ. Sau khi sửa cấu hình, khởi động lại:

```bash
docker compose restart fl-server
```

Mỗi lần bấm bắt đầu tạo một thư mục experiment riêng trong Docker volume `fl-output`, gồm:

- `rapport_federated_learning.csv`
- `model_federated_final.pt` sau round cuối

Sao chép kết quả từ container ra thư mục hiện tại:

```bash
docker compose cp fl-server:/app/server/runs ./server-runs
```

## 3. Kết nối client

Trên từng máy client, giữ dataset và môi trường huấn luyện như hiện tại rồi chạy:

```bash
python3 client/main_jetson.py \
  --data /duong-dan/toi/data.yaml \
  --server TEN_MIEN_HOAC_IP_VPS:8080 \
  --device-type jetson \
  --device-id js1
```

`device-id` phải khớp khóa trong `server/config_server.json`, hoặc client sẽ dùng cấu hình `default`.

## 4. Vận hành

```bash
docker compose ps
docker compose logs --tail=100 fl-server
docker compose stop
docker compose up -d
```

Không chạy `docker compose down -v` nếu chưa sao lưu kết quả. `docker compose down` thông thường vẫn giữ volume kết quả.

## Bảo mật mạng

Flower 1.4 trong mã hiện tại dùng gRPC không mã hóa. Không nên để cổng `8080` mở cho toàn Internet. Cách triển khai phù hợp nhất là cho VPS và các client vào cùng mạng riêng WireGuard/Tailscale, rồi firewall chỉ cho phép IP của mạng đó. Nếu buộc phải dùng IP công khai, giới hạn source IP của từng client trong firewall/security group; TLS cần được bổ sung đồng thời ở cả server và client.

Dashboard có HTTP Basic Auth nhưng chưa có HTTPS. Chỉ mở cổng `5000` qua VPN/Tailscale hoặc đặt dashboard sau reverse proxy HTTPS. Tuyệt đối đổi mật khẩu mặc định `change-me`.
