# tests/test_cell_modifier.py

"""
Test complet du CellModifier avec PDKManager et NetlistGenerator.

Teste:
- Génération de netlist modifiable
- Chargement et parsing des transistors
- Modification des largeurs W
- Sauvegarde et vérification
"""

import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator
from src.optimization.cell_modifier import CellModifier


def test_cell_modification():
    """Test principal de modification de cellule"""
    
    print("🧪 Test de CellModifier avec PDKManager\n")
    print("=" * 60)
    
    # ===== ÉTAPE 1: Initialiser le PDK =====
    print("\n📦 Initialisation du PDK...")
    try:
        pdk = PDKManager("sky130")
        print(f"✅ PDK chargé: {pdk.pdk_root}")
    except Exception as e:
        print(f"❌ Erreur PDK: {e}")
        return False
    
    # ===== ÉTAPE 2: Générer une netlist modifiable =====
    print("\n📝 Génération de la netlist...")
    gen = NetlistGenerator(pdk)
    
    output_path = "/tmp/inv_test.sp"
    
    try:
        netlist_path = gen.generate_characterization_netlist(
            cell_name="sky130_fd_sc_hd__inv_1",
            test_type="delay",
            output_path=output_path
        )
        print(f"✅ Netlist générée: {netlist_path}")
    except Exception as e:
        print(f"❌ Erreur génération: {e}")
        return False
    
    # ===== ÉTAPE 3: Charger avec CellModifier =====
    print("\n🔧 Chargement avec CellModifier...")
    try:
        modifier = CellModifier(netlist_path)
        print("✅ Netlist chargée")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return False
    
    # ===== ÉTAPE 4: Afficher l'état initial =====
    print("\n" + "=" * 60)
    print("📊 ÉTAT INITIAL")
    print("=" * 60)
    print(modifier.get_modification_summary())
    
    widths_initial = modifier.get_transistor_widths()
    print(f"\n🔍 Largeurs initiales: {widths_initial}")
    
    # Vérifier les valeurs attendues
    assert 'X0' in widths_initial, "❌ X0 (NFET) introuvable"
    assert 'X1' in widths_initial, "❌ X1 (PFET) introuvable"
    assert abs(widths_initial['X0'] - 650.0) < 1.0, f"❌ X0 devrait être 650nm, obtenu {widths_initial['X0']}"
    assert abs(widths_initial['X1'] - 1000.0) < 1.0, f"❌ X1 devrait être 1000nm, obtenu {widths_initial['X1']}"
    print("✅ Valeurs initiales correctes")
    
    # ===== ÉTAPE 5: Modifier les largeurs =====
    print("\n" + "=" * 60)
    print("⚙️  MODIFICATION DES LARGEURS")
    print("=" * 60)
    
    try:
        modifier.modify_width('X0', 700.0)   # NFET: 650 → 700nm
        print("✅ X0 modifié: 650nm → 700nm")
        
        modifier.modify_width('X1', 1200.0)  # PFET: 1000 → 1200nm
        print("✅ X1 modifié: 1000nm → 1200nm")
    except Exception as e:
        print(f"❌ Erreur modification: {e}")
        return False
    
    # Vérifier les nouvelles valeurs en mémoire
    widths_modified = modifier.get_transistor_widths()
    print(f"\n🔍 Largeurs modifiées (en mémoire): {widths_modified}")
    
    assert abs(widths_modified['X0'] - 700.0) < 1.0, f"❌ X0 devrait être 700nm, obtenu {widths_modified['X0']}"
    assert abs(widths_modified['X1'] - 1200.0) < 1.0, f"❌ X1 devrait être 1200nm, obtenu {widths_modified['X1']}"
    print("✅ Modifications en mémoire correctes")
    
    # ===== ÉTAPE 6: Sauvegarder =====
    print("\n" + "=" * 60)
    print("💾 SAUVEGARDE")
    print("=" * 60)
    
    output_modified = "/tmp/inv_modified.sp"
    
    try:
        saved_path = modifier.apply_modifications(output_modified)
        print(f"✅ Netlist sauvegardée: {saved_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return False
    
    # ===== ÉTAPE 7: Vérifier le fichier sauvegardé =====
    print("\n" + "=" * 60)
    print("🔍 VÉRIFICATION DU FICHIER")
    print("=" * 60)
    
    try:
        with open(output_modified, 'r') as f:
            content = f.read()
            
            print("\n📄 Lignes de transistors modifiées:")
            for line in content.split('\n'):
                if line.strip().startswith('X') and 'sky130_fd_pr__' in line:
                    print(f"  {line.strip()}")
            
            # Vérifications automatiques
            print("\n🧪 Tests de validation:")
            
            # Test 1: X0 doit avoir w=700000u
            if 'X0' in content and 'w=700000u' in content:
                print("  ✅ X0 correctement sauvegardé (w=700000u)")
            else:
                print("  ❌ X0 non trouvé ou mal formaté")
                return False
            
            # Test 2: X1 doit avoir w=1200000u
            if 'X1' in content and 'w=1200000u' in content:
                print("  ✅ X1 correctement sauvegardé (w=1200000u)")
            else:
                print("  ❌ X1 non trouvé ou mal formaté")
                return False
            
            # Test 3: Longueurs inchangées
            if content.count('l=150000u') >= 2:
                print("  ✅ Longueurs L préservées (l=150000u)")
            else:
                print("  ❌ Longueurs L modifiées par erreur")
                return False
    
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        return False
    
    # ===== RÉSUMÉ FINAL =====
    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS PASSENT")
    print("=" * 60)
    print(f"""
📊 Résumé:
  • Netlist générée: {netlist_path}
  • Netlist modifiée: {output_modified}
  • X0 (NFET): 650nm → 700nm ✅
  • X1 (PFET): 1000nm → 1200nm ✅
  • Longueurs L: inchangées ✅
    """)
    
    return True


if __name__ == "__main__":
    success = test_cell_modification()
    sys.exit(0 if success else 1)
