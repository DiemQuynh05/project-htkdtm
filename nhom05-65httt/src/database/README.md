# Cấu hình cơ sở dữ liệu

Hệ thống khuyến nghị sử dụng PostgreSQL 16.x làm cơ sở dữ liệu định hướng.

Môi trường phát triển có thể dùng SQLite thông qua `DB_ENGINE=sqlite` trong `.env`.
Môi trường PostgreSQL sử dụng:

- Host: 127.0.0.1
- Port: 5432
- Database: training_analytics
- Driver Python: psycopg 3.2.12

Tạo database:

```sql
CREATE DATABASE training_analytics
    WITH ENCODING 'UTF8';
```

Cập nhật các biến `DB_*` trong `.env` rồi chạy migrations bằng Django.
