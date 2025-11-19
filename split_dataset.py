import os
import shutil
import random

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
BASE_PATH = "emotion_dataset" 

# Ratios de división recomendados para Deep Learning
TRAIN_RATIO = 0.70  # 70% para Entrenamiento
VAL_RATIO   = 0.15  # 15% para Validación
TEST_RATIO  = 0.15  # 15% para Prueba (Total: 100%)

CLASSES = ["angry", "happy", "neutral", "sad", "surprise"]
SETS = ["train", "validation", "test"]


# --- FUNCIÓN DE DIVISIÓN PRINCIPAL ---

def split_dataset(base_path, classes, train_ratio, val_ratio, test_ratio):
    """Divide las imágenes de cada clase en los directorios de Train, Val y Test."""
    print("--- INICIANDO LA DIVISIÓN 70/15/15 ---")
    

    set_paths = {
        "train": os.path.join(base_path, "train"),
        "validation": os.path.join(base_path, "validation"),
        "test": os.path.join(base_path, "test")
    }

    # 1. Limpiar y crear la nueva estructura de sets
    for path in set_paths.values():
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        for cls in classes:
            os.makedirs(os.path.join(path, cls))

    print("Estructura de carpetas limpia y lista.")

    # 2. Asignar y mover archivos
    total_moved = 0
    
    for cls in classes:
        original_class_path = os.path.join(base_path, cls)
        
        if not os.path.isdir(original_class_path):
            print(f"ERROR: No se encontró la carpeta original de la clase: {cls}. Verifique la ruta.")
            continue
        
        all_files = [f for f in os.listdir(original_class_path) if os.path.isfile(os.path.join(original_class_path, f))]
        random.shuffle(all_files) # Mezclar para asegurar una división aleatoria
        
        total_files = len(all_files)
        
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]

        # 3. Mover a las carpetas de destino
        file_distribution = {
            "train": train_files, 
            "validation": val_files, 
            "test": test_files
        }

        for set_name, files in file_distribution.items():
            dest_path = os.path.join(set_paths[set_name], cls)
            for filename in files:
                src = os.path.join(original_class_path, filename)
                dst = os.path.join(dest_path, filename)
                shutil.move(src, dst)
            total_moved += len(files)
        
        print(f"  - Clase '{cls}': Total procesado: {total_files} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    print(f"\nDivisión completada. Total de archivos movidos: {total_moved}")
    print("--------------------------------------------------")


# --- FUNCIÓN DE VERIFICACIÓN ---

def count_images_in_sets(base_path, sets, classes):
    """Cuenta y muestra el número de imágenes por clase en cada set de datos."""
    print("\n--- VERIFICACIÓN DEL CONTEO FINAL (Train, Validation, Test) ---")
    
    grand_total = 0
    
    for set_name in sets:
        set_path = os.path.join(base_path, set_name)
        total_set_count = 0
        
        print(f"\n## SET: {set_name.upper()}")
        
        for cls in classes:
            class_path = os.path.join(set_path, cls)
            
            if os.path.isdir(class_path):
                num_files = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
                total_set_count += num_files
                print(f"  - {cls.capitalize()}: {num_files}")
            else:
                print(f"  - {cls.capitalize()}: Carpeta no encontrada.")
        
        print(f"  -> Total de imágenes en {set_name.upper()}: {total_set_count}")
        grand_total += total_set_count

    print(f"\n--- Conteo Finalizado. Gran Total de imágenes: {grand_total} ---")


# --- EJECUCIÓN DEL PROGRAMA ---

if __name__ == "__main__":
    
    # 1. Ejecutar la división
    split_dataset(BASE_PATH, CLASSES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # 2. Ejecutar la verificación de conteo
    count_images_in_sets(BASE_PATH, SETS, CLASSES)