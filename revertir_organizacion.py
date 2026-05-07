import os
import shutil

def revert():
    # Carpetas donde se organizó todo
    folders = ['src', 'scripts_r', 'docs', 'models', 'data', 'results', 'media', 'data_samples']
    
    print("[*] Iniciando reversión de carpetas a la raíz...")
    
    for folder in folders:
        if os.path.exists(folder):
            print(f"[*] Procesando carpeta: {folder}")
            # Listar todo dentro de la carpeta
            items = os.listdir(folder)
            for item in items:
                source = os.path.join(folder, item)
                target = os.path.join('.', item) # Raíz
                
                # Si el destino ya existe, agregamos un sufijo o lo ignoramos si es el mismo
                if os.path.exists(target):
                    if os.path.isdir(target) and os.path.isdir(source):
                        # Si ambos son carpetas, intentamos mover el contenido
                        for sub_item in os.listdir(source):
                            sub_source = os.path.join(source, sub_item)
                            sub_target = os.path.join(target, sub_item)
                            if not os.path.exists(sub_target):
                                try:
                                    shutil.move(sub_source, sub_target)
                                except: pass
                        continue
                    else:
                        print(f"    [!] El archivo {item} ya existe en la raíz. Saltando...")
                        continue
                
                try:
                    shutil.move(source, target)
                    print(f"    [OK] {item} -> Raíz/")
                except Exception as e:
                    print(f"    [ERR] No se pudo mover {item}: {e}")

    print("\n[!] REVERSIÓN FINALIZADA. Todos los archivos han vuelto a la raíz.")

if __name__ == "__main__":
    revert()
