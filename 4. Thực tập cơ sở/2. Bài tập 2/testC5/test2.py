from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# Tải và tiền xử lý dữ liệu
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

# Visualize phân bố nhãn bằng histogram
plt.figure(figsize=(10, 6))  # Kích thước biểu đồ
plt.hist(train_labels, bins=10, range=(-0.5, 9.5), rwidth=0.8, color='skyblue', edgecolor='black')
plt.title("Phân bố nhãn trong tập dữ liệu MNIST", fontsize=14)
plt.xlabel("Nhãn (Chữ số từ 0-9)", fontsize=12)
plt.ylabel("Số lượng mẫu", fontsize=12)
plt.xticks(range(10))  # Đặt nhãn trục x là 0-9
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Thêm lưới cho dễ nhìn
plt.show()