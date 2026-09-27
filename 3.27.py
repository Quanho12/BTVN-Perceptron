import numpy as np
w = np.array([1,2,-10])
x = np.array([3,4,1])
y_true = -1
wTx = np.dot(w,x)
print(f"1. Gia tri w^Tx * x = {wTx}")
y_pred = 1 if wTx >=0 else -1
print(f"2. Nhan du doan cua mo hinh: {y_pred}")
print(f" Nhan thuc te: {y_true}")
if y_pred != y_true:
    print("3. Ket luan: Diem du lieu bi phan lop sai")
    w_new = w + y_true *x
    print(f"   => Tiến hành cập nhật trọng số.")
    print(f"   => Trọng số w mới: {w_new}")
else:
    print("3. Kết luận: Điểm dữ liệu được phân lớp đúng!")
