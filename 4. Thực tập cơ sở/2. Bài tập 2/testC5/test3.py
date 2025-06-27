from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Tải và tiền xử lý dữ liệu
(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

# Thêm nhiễu ngẫu nhiên
train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 784))], axis=1)

# Thêm số 0
train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))], axis=1)

# Hàm tạo mô hình với learning rate tùy chỉnh
def get_model(learning_rate=0.001):  # Thêm tham số learning_rate, mặc định là 0.001
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    # Tạo optimizer RMSprop với learning rate tùy chỉnh
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Huấn luyện với dữ liệu chứa nhiễu
model_noise = get_model(learning_rate=0.001)  # Thay đổi learning rate tại đây
history_noise = model_noise.fit(
    train_images_with_noise_channels, train_labels,
    epochs=30,
    batch_size=128,
    validation_split=0.2)

# Huấn luyện với dữ liệu chứa số 0
model_zeros = get_model(learning_rate=0.001)  # Thay đổi learning rate tại đây
history_zeros = model_zeros.fit(
    train_images_with_zeros_channels, train_labels,
    epochs=30,
    batch_size=128,
    validation_split=0.2)

# Vẽ biểu đồ
val_acc_noise = history_noise.history["val_accuracy"]
val_acc_zeros = history_zeros.history["val_accuracy"]
epochs = range(1, 31)  # Sửa lại epochs từ 1-11 thành 1-31 vì epochs=30

plt.plot(epochs, val_acc_noise, "b-", label="Validation accuracy with noise channels")
plt.plot(epochs, val_acc_zeros, "b--", label="Validation accuracy with zeros channels")
plt.title("Effect of noise channels on validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()