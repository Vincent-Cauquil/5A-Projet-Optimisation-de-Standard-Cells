"""
Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026
"""
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
from src.gui.app_main import MainWindow

def main():
    # Création de l'application
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()