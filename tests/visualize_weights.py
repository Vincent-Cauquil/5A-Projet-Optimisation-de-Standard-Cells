# scripts/visualize_weights.py
#!/usr/bin/env python3
"""
Visualise les poids sauvegardés
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.weight_manager import WeightManager
import matplotlib.pyplot as plt
import numpy as np

def plot_category_performance(weight_manager: WeightManager, category: str):
    """Plot des performances d'une catégorie"""
    summary = weight_manager.get_category_summary(category)
    
    if not summary:
        print(f"⚠️  Pas de données pour {category}")
        return
    
    cells = summary['cells']
    
    cell_names = list(cells.keys())
    delays = [c['metrics']['delay_avg_ps'] for c in cells.values()]
    powers = [c['metrics']['power_avg_uw'] for c in cells.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Délais
    ax1.bar(cell_names, delays, color='skyblue')
    ax1.set_title(f'Délais - Catégorie {category.upper()}')
    ax1.set_ylabel('Delay (ps)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Puissance
    ax2.bar(cell_names, powers, color='salmon')
    ax2.set_title(f'Puissance - Catégorie {category.upper()}')
    ax2.set_ylabel('Power (µW)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'results_{category}.png', dpi=150)
    print(f"✅ Graphique sauvegardé: results_{category}.png")

def main():
    wm = WeightManager()
    
    print("📊 Visualisation des poids optimisés\n")
    
    # Lister toutes les catégories
    categories = list(set(wm._get_category(c) for c in wm.list_available_cells()))
    
    for cat in categories:
        print(f"\n{'='*60}")
        print(f"Catégorie: {cat.upper()}")
        print('='*60)
        
        summary = wm.get_category_summary(cat)
        if summary and 'statistics' in summary:
            stats = summary['statistics']
            print(f"  Nombre de cellules: {stats['n_cells']}")
            print(f"  Délai moyen: {stats['avg_delay_ps']:.2f} ps")
            print(f"  Puissance moyenne: {stats['avg_power_uw']:.2f} µW")
        
        plot_category_performance(wm, cat)

if __name__ == "__main__":
    main()
