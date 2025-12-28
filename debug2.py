from pathlib import Path

def fix_line_endings():
    # Le chemin exact qui pose problème
    target_file = Path("netlists/templates/rc_filter.cir").resolve()
    
    if not target_file.exists():
        print("Fichier introuvable !")
        return

    print(f"Nettoyage de : {target_file}")
    
    # Lecture du contenu
    try:
        content = target_file.read_bytes()
        # Remplacement forcé des sauts de ligne Windows (\r\n) par Unix (\n)
        new_content = content.replace(b'\r\n', b'\n')
        
        # On sauvegarde par dessus
        target_file.write_bytes(new_content)
        print("✅ Fichier converti en format Unix (LF).")
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    fix_line_endings()