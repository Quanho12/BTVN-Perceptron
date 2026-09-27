import numpy as np

class Perceptron:
    def __init__(self, max_iter=100):
        # Khai báo số vòng lặp tối đa để tránh máy tính chạy vô tận
        self.max_iter = max_iter
        # Biến lưu trữ ma trận trọng số (sẽ được tạo ra khi gọi hàm fit)
        self.w = None
        
    def fit(self, X, y):
        # 1. Tự động thêm cột bias (cột toàn số 1) vào vị trí đầu tiên của ma trận X
        num_samples = X.shape[0]
        X_bias = np.c_[np.ones((num_samples, 1)), X]
        
        # 2. Khởi tạo trọng số ban đầu là một vector toàn số 0
        num_features = X_bias.shape[1]
        self.w = np.zeros(num_features)
        
        # 3. Bắt đầu vòng lặp huấn luyện
        for epoch in range(self.max_iter):
            has_error = False
            
            # Duyệt qua từng điểm dữ liệu trong tập X_bias
            for i in range(num_samples):
                x_i = X_bias[i]
                y_i = y[i]
                
                # Nếu mẫu dữ liệu bị phân lớp sai (trái dấu nhau nên tích <= 0)
                if y_i * np.dot(self.w, x_i) <= 0:
                    # Áp dụng công thức cập nhật Perceptron: w_new = w + y_i * x_i
                    self.w = self.w + y_i * x_i
                    has_error = True
            
            # Nếu duyệt qua toàn bộ dữ liệu mà không có điểm nào sai, thuật toán đã tìm được nghiệm
            if not has_error:
                print(f"Thuật toán hội tụ thành công sau {epoch + 1} vòng lặp!")
                break
                
    def predict(self, X):
        # Tự động thêm cột bias cho tập dữ liệu mới
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Tính w^T * X cho toàn bộ điểm dữ liệu
        wTx = np.dot(X_bias, self.w)
        
        # Hàm sgn: trả về 1 nếu giá trị >= 0, ngược lại trả về -1
        y_pred = np.where(wTx >= 0, 1, -1)
        return y_pred