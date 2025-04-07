import tensorflow as tf
import time
import os
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Загрузка датасета
dataset_path = kagglehub.dataset_download("sayangupta001/mnist-greek-letters")


# Определяем генератор данных с аугментацией
datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # Нормализация изображений
    horizontal_flip=True,        # Горизонтальное отражение
    rotation_range=20,           # Вращение на случайные углы от -20 до 20 градусов
    zoom_range=0.2              # Увеличение/уменьшение изображений
)

# Загрузка данных и применение аугментации
train_generator = datagen.flow_from_directory(
    dataset_path + "/train",     # Путь к директории с изображениями
    target_size=(28, 28),        # Размер изображений (оставляем 28x28)
    batch_size=24,               # Размер пакета
    class_mode='categorical',    # Многоклассовая классификация
    color_mode='grayscale'       # Указываем, что изображения черно-белые
)


# Получаем несколько примеров аугментированных изображений
def visualize_augmentation(generator, num_images=5):
    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        # Получаем один батч из генератора
        images, labels = next(generator)

        # Отображаем несколько изображений
        for j in range(min(images.shape[0], 5)):  # Показываем первые 5 изображений в батче
            plt.subplot(1, 5, j + 1)
            plt.imshow(images[j].squeeze(),
                       cmap='gray')  # Поскольку изображения ч/б, используем 'gray' для правильного отображения
            plt.axis('off')

        plt.show()

def measure_time(num_workers):
    start_time = time.time()

    dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),   # Изображения размером 28x28 с 1 каналом
            tf.TensorSpec(shape=(None, len(train_generator.class_indices)), dtype=tf.float32)  # Многоклассовые метки
        )
    )

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(24,drop_remainder=True)

    if num_workers == 1:
        dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=1)  # Одноядерная обработка
    else:
        dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)  # Многоядерная обработка

    # Обработка 100 батчей
    for batch in dataset.take(100):  # Обработка 100 батчей
        pass

    end_time = time.time()
    return end_time - start_time

# Запуск и сравнение времени выполнения
single_core_time = measure_time(num_workers=1)
multi_core_time = measure_time(num_workers=tf.data.AUTOTUNE)
visualize_augmentation(train_generator, num_images=1)

print(f"Single-core processing time: {single_core_time:.2f} seconds")
print(f"Multi-core processing time: {multi_core_time:.2f} seconds")
