import numpy as np 
w = np.array([-2 ,1, 0])
x = np.array([2,3,1])
y_true = 1
wTX = np.dot(w, x)
print(f"Gia tri w^T * x ban dau = {wTX}")
y_pred = 1 if wTX >= 0 else 0
print(f" Nhan du doan: {y_pred}")
print(f" Nhan thuc te: {y_true}")
if y_pred != y_true:
    print(f"Diem du lieu bi phan lop sai\n")
    w_new = w +y_true *x
    print(f"   => Tiến hành cập nhật trọng số.")
    print(f"   => Trọng số w mới: {w_new}")
    wTx_new = np.dot(w_new, x)
    print(f"Giá trị w^T * x sau khi cập nhật = {wTx_new}")
    
    y_pred_new = 1 if wTx_new >= 0 else -1
    print(f"Nhãn dự đoán mới: {y_pred_new} (Đã sửa lỗi thành công!)")
else:
    print("=> Kết luận: Điểm dữ liệu được phân lớp đúng, không cần cập nhật.")