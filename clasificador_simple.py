# clasificador_simple.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print("🚀 CLASIFICADOR SIMPLE DE DEPORTES")

# Configuración
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20

# Ruta de datos
data_dir = "sportimages/sportimages"

# Verificar que existe
if not os.path.exists(data_dir):
    print(f"❌ Error: No existe {data_dir}")
    exit()

# Generadores de datos
print("📁 Creando generadores...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% para validación
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"✅ Datos cargados:")
print(f"   Clases: {list(train_generator.class_indices.keys())}")
print(f"   Entrenamiento: {train_generator.samples} imágenes")
print(f"   Validación: {val_generator.samples} imágenes")

# Modelo simple
print("🧠 Creando modelo...")

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
print("🎯 Entrenando modelo...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)

# Resultados
print("📊 Graficando resultados...")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.legend()

plt.tight_layout()
plt.savefig('resultados.png', dpi=150)
plt.show()

# Guardar modelo
model.save('modelo_simple_deportes.h5')
print("💾 Modelo guardado como 'modelo_simple_deportes.h5'")

print(f"✅ ¡Listo! Modelo entrenado para {len(train_generator.class_indices)} deportes")