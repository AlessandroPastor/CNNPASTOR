# clasificador_deportes_fix.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import glob

# Configuración
IMG_ALTURA = 150
IMG_ANCHURA = 150
BATCH_SIZE = 32
EPOCHS = 30

# Ruta específica para tu estructura
RUTA_BASE = "sportimages/sportimages"

def obtener_estadisticas_datos():
    """
    Obtiene estadísticas sin cargar todas las imágenes en memoria
    """
    print("=== ANALIZANDO DATOS ===\n")
    
    nombres_clases = []
    conteo_imagenes = []
    
    # Obtener carpetas de deportes
    carpetas_deportes = [d for d in os.listdir(RUTA_BASE) 
                        if os.path.isdir(os.path.join(RUTA_BASE, d)) and not d.startswith('_')]
    carpetas_deportes.sort()
    
    print(f"Deportes encontrados: {carpetas_deportes}")
    
    for deporte in carpetas_deportes:
        ruta_deporte = os.path.join(RUTA_BASE, deporte)
        
        # Contar imágenes JPG/JPEG
        archivos_jpg = glob.glob(os.path.join(ruta_deporte, "*.jpg"))
        archivos_jpg.extend(glob.glob(os.path.join(ruta_deporte, "*.JPG")))
        archivos_jpg.extend(glob.glob(os.path.join(ruta_deporte, "*.jpeg")))
        archivos_jpg.extend(glob.glob(os.path.join(ruta_deporte, "*.JPEG")))
        
        nombres_clases.append(deporte)
        conteo_imagenes.append(len(archivos_jpg))
        
        print(f"  {deporte}: {len(archivos_jpg)} imágenes")
    
    total_imagenes = sum(conteo_imagenes)
    print(f"\nTotal de imágenes: {total_imagenes}")
    
    return nombres_clases, conteo_imagenes, total_imagenes

def crear_generadores_datos():
    """
    Crea generadores de datos que cargan imágenes por lotes
    """
    print("\n=== CREANDO GENERADORES DE DATOS ===\n")
    
    # Data augmentation para entrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode='nearest',
        validation_split=0.2  # 20% para validación
    )
    
    # Solo rescale para test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        RUTA_BASE,
        target_size=(IMG_ALTURA, IMG_ANCHURA),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Generador de validación
    validation_generator = train_datagen.flow_from_directory(
        RUTA_BASE,
        target_size=(IMG_ALTURA, IMG_ANCHURA),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    print(f"\n✅ Generadores creados exitosamente")
    print(f"   Clases: {list(train_generator.class_indices.keys())}")
    print(f"   Imágenes de entrenamiento: {train_generator.samples}")
    print(f"   Imágenes de validación: {validation_generator.samples}")
    
    return train_generator, validation_generator

def crear_modelo_cnn(num_clases):
    """
    Crea modelo CNN optimizado
    """
    model = Sequential()
    
    # Capas convolucionales
    model.add(Conv2D(32, (3, 3), activation='relu', 
                     input_shape=(IMG_ALTURA, IMG_ANCHURA, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Capas densas
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_clases, activation='softmax'))
    
    # Compilar
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def visualizar_lote_ejemplo(generator):
    """
    Visualiza un lote de imágenes del generador
    """
    print("\n🖼️  VISUALIZANDO EJEMPLOS...")
    
    # Obtener un lote
    x_batch, y_batch = next(generator)
    
    # Nombres de clases
    class_names = list(generator.class_indices.keys())
    
    # Visualizar
    plt.figure(figsize=(12, 8))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(x_batch[i])
        
        # Obtener clase real
        class_idx = np.argmax(y_batch[i])
        plt.title(f'{class_names[class_idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('ejemplos_deportes.png', dpi=150, bbox_inches='tight')
    plt.show()

def entrenar_con_generadores(train_gen, val_gen, num_clases):
    """
    Entrena el modelo usando generadores
    """
    # Crear modelo
    model = crear_modelo_cnn(num_clases)
    
    # Callbacks para mejor entrenamiento
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'mejor_modelo_deportes.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n=== INICIANDO ENTRENAMIENTO ===")
    print(f"Steps por época: {train_gen.samples // BATCH_SIZE}")
    print(f"Validation steps: {val_gen.samples // BATCH_SIZE}")
    
    # Entrenamiento
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def graficar_resultados(history):
    """
    Grafica los resultados del entrenamiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precisión
    ax1.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    ax1.set_title('Precisión del Modelo', fontsize=14)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Precisión', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pérdida
    ax2.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validación', linewidth=2)
    ax2.set_title('Pérdida del Modelo', fontsize=14)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Pérdida', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('resultados_entrenamiento.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Función principal optimizada para grandes datasets
    """
    print("🏀⚽🎾 CLASIFICADOR DE DEPORTES (OPTIMIZADO) 🏈⚾🏒\n")
    
    # 1. Obtener estadísticas
    nombres_clases, conteos, total = obtener_estadisticas_datos()
    
    if total == 0:
        print("❌ No se encontraron imágenes")
        return
    
    num_clases = len(nombres_clases)
    
    # 2. Crear generadores
    train_generator, validation_generator = crear_generadores_datos()
    
    # 3. Visualizar ejemplos
    visualizar_lote_ejemplo(train_generator)
    
    # 4. Entrenar modelo
    history, model = entrenar_con_generadores(
        train_generator, validation_generator, num_clases
    )
    
    # 5. Resultados
    graficar_resultados(history)
    
    # Evaluar modelo final
    print("\n=== EVALUACIÓN FINAL ===")
    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"🎯 Precisión final en validación: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📉 Pérdida final: {loss:.4f}")
    
    # Guardar modelo final
    model.save('modelo_deportes_final.h5')
    print(f"\n💾 Modelo guardado como: 'modelo_deportes_final.h5'")
    print("💾 Mejor modelo guardado como: 'mejor_modelo_deportes.h5'")
    
    print(f"\n✅ ENTRENAMIENTO COMPLETADO!")
    print(f"   Deportes clasificados: {nombres_clases}")

if __name__ == "__main__":
    main()