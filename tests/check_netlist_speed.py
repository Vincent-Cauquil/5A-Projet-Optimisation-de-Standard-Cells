# tests/check_netlist_speed.py
#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig

def main():
    pdk = PDKManager("sky130", verbose=False)
    config = SimulationConfig(vdd=1.8, temp=27, corner='tt')
    
    gen = NetlistGenerator(pdk)
    
    # Générer une netlist
    output = "/tmp/test_speed.sp"
    gen.generate_characterization_netlist(
        cell_name='sky130_fd_sc_hd__inv_1',
        output_path=output,
        config=config
    )
    
    # Lire et analyser
    with open(output) as f:
        content = f.read()
    
    print("="*60)
    print("🔍 ANALYSE DE LA NETLIST GÉNÉRÉE")
    print("="*60)
    
    # Trouver la ligne .tran
    for line in content.split('\n'):
        if '.tran' in line.lower():
            print(f"\n📊 Ligne trouvée: {line}")
            
            if '1p' in line:
                print("❌ Pas de temps = 1ps (LENT)")
                print("   → Modifier à 10p dans netlist_generator.py")
            elif '10p' in line:
                print("✅ Pas de temps = 10ps (RAPIDE)")
            
            # Extraire la durée totale
            parts = line.split()
            if len(parts) >= 3:
                duration = parts[2]
                print(f"   Durée totale: {duration}")
                
                # Calculer nombre de points
                if '1p' in line:
                    factor = 1e-12
                elif '10p' in line:
                    factor = 10e-12
                else:
                    factor = None
                
                if factor:
                    duration_val = float(duration.rstrip('n')) * 1e-9
                    n_points = int(duration_val / factor)
                    print(f"   Nombre de points: {n_points:,}")
                    
                    if n_points > 1000:
                        print(f"   ⚠️  Trop de points ! (~{n_points/1000:.1f}s de simulation)")
                    else:
                        print(f"   ✅ OK (~{n_points/1000:.1f}s de simulation)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
