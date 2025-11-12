import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
import pandas as pd
from datetime import datetime

# Configuración para mejores gráficos
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [10, 6]

print("=" * 60)
print("SISTEMA DE CLASIFICACIÓN DE IMÁGENES DEPORTIVAS")
print("=" * 60)

# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================
print("\n📁 FASE 1: CARGA DE IMÁGENES")
print("-" * 40)

# Ruta base del conjunto de imágenes
base_dir = os.path.join(os.getcwd(), "sportimages")

# Listas para almacenar información
images = []
directories = []
dircount = []

print(f"Leyendo imágenes desde: {base_dir}\n")

cant = 0
prev_root = None

# Recorre recursivamente todas las carpetas e imágenes
for root, _, filenames in os.walk(base_dir):
    image_files = [f for f in filenames if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", f, re.IGNORECASE)]
    
    if image_files:
        directories.append(root)
        count_in_dir = 0
        
        for filename in image_files:
            filepath = os.path.join(root, filename)
            try:
                image = plt.imread(filepath)
                images.append(image)
                count_in_dir += 1
                cant += 1
                print(f"📸 Leyendo imagen {cant}: {filename}", end="\r")
            except Exception as e:
                print(f"\n⚠️ Error leyendo {filepath}: {e}")
        
        dircount.append(count_in_dir)
        print(f"\n📂 {root} → {count_in_dir} imágenes")

# Resumen final
print("\n📊 RESUMEN DE CARGA:")
print(f"   • Directorios leídos: {len(directories)}")
print(f"   • Imágenes por directorio: {dircount}")
print(f"   • Total de imágenes: {sum(dircount)}")

# =============================================================================
# 2. CREACIÓN DE ETIQUETAS Y METADATOS
# =============================================================================
print("\n🏷️ FASE 2: CREACIÓN DE ETIQUETAS")
print("-" * 40)

# CREAR LAS ETIQUETAS QUE FALTABAN
labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1

print(f"✅ Etiquetas creadas: {len(labels)}")

deportes = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    deporte_nombre = name[len(name)-1]
    print(f"   🎯 Clase {indice}: {deporte_nombre}")
    deportes.append(deporte_nombre)
    indice = indice + 1

# Convertir a arrays de numpy
y = np.array(labels)
X = np.array(images, dtype=np.uint8)

# Información de clases
classes = np.unique(y)
nClasses = len(classes)
print(f'\n🎯 INFORMACIÓN DE CLASES:')
print(f'   • Número de clases: {nClasses}')
print(f'   • Clases: {classes}')

# =============================================================================
# 3. PREPARACIÓN DE DATOS
# =============================================================================
print("\n🔧 FASE 3: PREPARACIÓN DE DATOS")
print("-" * 40)

# Dividir los datos
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42, stratify=train_y)

# Normalizar los datos
train_X = train_X.astype('float32') / 255.0
valid_X = valid_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0

# Convertir etiquetas a one-hot encoding
train_label = to_categorical(train_y, nClasses)
valid_label = to_categorical(valid_y, nClasses)
test_Y_one_hot = to_categorical(test_y, nClasses)

print(f"📐 DIMENSIONES:")
print(f"   • Entrenamiento: {train_X.shape} -> {train_label.shape}")
print(f"   • Validación: {valid_X.shape} -> {valid_label.shape}")
print(f"   • Prueba: {test_X.shape} -> {test_Y_one_hot.shape}")

# =============================================================================
# 4. VISUALIZACIÓN DE DATOS
# =============================================================================
print("\n📊 FASE 4: VISUALIZACIÓN DE DATOS")
print("-" * 40)

# Gráfico de distribución de clases
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
class_distribution = [np.sum(train_y == i) for i in range(nClasses)]
bars = plt.bar(range(nClasses), class_distribution, color=plt.cm.Set3(np.linspace(0, 1, nClasses)))
plt.title('DISTRIBUCIÓN DE CLASES - ENTRENAMIENTO', fontweight='bold')
plt.xlabel('Clases')
plt.ylabel('Número de Imágenes')
plt.xticks(range(nClasses), [f'C{i}' for i in range(nClasses)])

# Añadir valores en las barras
for bar, count in zip(bars, class_distribution):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(count), ha='center', va='bottom')

# Ejemplos de imágenes
plt.subplot(1, 2, 2)
if len(train_X) >= 6:
    # Mostrar primeras 6 imágenes
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(train_X[i])
        plt.title(f'Clase: {train_y[i]}')
        plt.axis('off')
    plt.suptitle('EJEMPLOS DE IMÁGENES', fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# 5. CONSTRUCCIÓN DEL MODELO CNN
# =============================================================================
print("\n🧠 FASE 5: CONSTRUCCIÓN DEL MODELO CNN")
print("-" * 40)

INIT_LR = 1e-3
epochs = 6
batch_size = 64

input_shape = train_X.shape[1:]
print(f"   • Input shape: {input_shape}")
print(f"   • Tasa de aprendizaje: {INIT_LR}")
print(f"   • Épocas: {epochs}")
print(f"   • Batch size: {batch_size}")

# Construir modelo
sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=input_shape))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5)) 
sport_model.add(Dense(nClasses, activation='softmax'))

sport_model.summary()

# Compilar modelo
sport_model.compile(
    loss=keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adagrad(learning_rate=INIT_LR), 
    metrics=['accuracy']
)

# =============================================================================
# 6. ENTRENAMIENTO DEL MODELO
# =============================================================================
print("\n🚀 FASE 6: ENTRENAMIENTO DEL MODELO")
print("-" * 40)

print("⏳ Iniciando entrenamiento...")
history = sport_model.fit(
    train_X, train_label, 
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

# =============================================================================
# 7. EVALUACIÓN DEL MODELO
# =============================================================================
print("\n📈 FASE 7: EVALUACIÓN DEL MODELO")
print("-" * 40)

# Guardar modelo
sport_model.save("sports_classifier.h5")
print("💾 Modelo guardado como 'sports_classifier.h5'")

# Evaluar modelo
test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print(f'\n🎯 RESULTADOS EN PRUEBA:')
print(f'   • Pérdida: {test_eval[0]:.4f}')
print(f'   • Precisión: {test_eval[1]:.4f} ({test_eval[1]*100:.2f}%)')

# Predicciones
predicted_classes = sport_model.predict(test_X, verbose=0)
predicted_classes = np.argmax(predicted_classes, axis=1)

# Reporte de clasificación
target_names = [f"{deportes[i] if i < len(deportes) else f'Clase {i}'}" for i in range(nClasses)]
print("\n📋 REPORTE DE CLASIFICACIÓN:")
print(classification_report(test_y, predicted_classes, target_names=target_names))

# =============================================================================
# 8. VISUALIZACIÓN DE RESULTADOS
# =============================================================================
print("\n📊 FASE 8: VISUALIZACIÓN DE RESULTADOS")
print("-" * 40)

H = history.history
acc_key = 'accuracy' if 'accuracy' in H else 'acc'
val_acc_key = 'val_accuracy' if 'val_accuracy' in H else 'val_acc'

epochs_range = range(1, len(H['loss']) + 1)

# Crear figura con múltiples subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. GRÁFICO DE PRECISIÓN
ax1.plot(epochs_range, H[acc_key], 'o-', linewidth=2, markersize=6, 
         label='Precisión Entrenamiento', color='#2E86AB')
ax1.plot(epochs_range, H[val_acc_key], 's-', linewidth=2, markersize=6, 
         label='Precisión Validación', color='#A23B72')
ax1.set_title('EVOLUCIÓN DE LA PRECISIÓN', fontsize=14, fontweight='bold')
ax1.set_xlabel('Época')
ax1.set_ylabel('Precisión')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# 2. GRÁFICO DE PÉRDIDA
ax2.plot(epochs_range, H['loss'], 'o-', linewidth=2, markersize=6, 
         label='Pérdida Entrenamiento', color='#F18F01')
ax2.plot(epochs_range, H['val_loss'], 's-', linewidth=2, markersize=6, 
         label='Pérdida Validación', color='#C73E1D')
ax2.set_title('EVOLUCIÓN DE LA PÉRDIDA', fontsize=14, fontweight='bold')
ax2.set_xlabel('Época')
ax2.set_ylabel('Pérdida')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. MATRIZ DE CONFUSIÓN
cm = confusion_matrix(test_y, predicted_classes)
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
ax3.set_title('MATRIZ DE CONFUSIÓN', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax3)

# Etiquetas de la matriz de confusión
tick_marks = np.arange(nClasses)
ax3.set_xticks(tick_marks)
ax3.set_yticks(tick_marks)
ax3.set_xticklabels([f'C{i}' for i in range(nClasses)])
ax3.set_yticklabels([f'C{i}' for i in range(nClasses)])

# Añadir valores en las celdas
thresh = cm.max() / 2.
for i in range(nClasses):
    for j in range(nClasses):
        ax3.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

ax3.set_ylabel('Etiqueta Real')
ax3.set_xlabel('Etiqueta Predicha')

# 4. MÉTRICAS PRINCIPALES
ax4.axis('off')
metrics_text = f"""
RESUMEN DEL MODELO

Precisión Final: {test_eval[1]:.4f}
Pérdida Final: {test_eval[0]:.4f}

Total Épocas: {epochs}
Batch Size: {batch_size}
Tasa Aprendizaje: {INIT_LR}

Clases: {nClasses}
Imágenes: {len(X)}
"""
ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('resultados_entrenamiento.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 9. REPORTE FINAL
# =============================================================================
print("\n" + "="*60)
print("INFORME FINAL DEL ENTRENAMIENTO")
print("="*60)

print(f"\n📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Datos procesados: {len(X)} imágenes, {nClasses} clases")
print(f"🎯 Precisión final: {test_eval[1]:.4f} ({test_eval[1]*100:.2f}%)")
print(f"📉 Pérdida final: {test_eval[0]:.4f}")

print(f"\n🏆 CLASIFICACIÓN POR RENDIMIENTO:")
if test_eval[1] >= 0.9:
    print("   ✅ EXCELENTE - Modelo de alto rendimiento")
elif test_eval[1] >= 0.7:
    print("   👍 BUENO - Modelo funcional")
elif test_eval[1] >= 0.5:
    print("   ⚠️  REGULAR - Podría necesitar mejoras")
else:
    print("   ❌ BAJO - Se recomienda revisar datos y modelo")

print(f"\n💾 Archivos generados:")
print(f"   • sports_classifier.h5 - Modelo entrenado")
print(f"   • resultados_entrenamiento.png - Gráficos de resultados")

print("\n" + "="*60)