# src/simulation/pool.py - Section optimisation NGSpice

class OptimizedNGSpiceConfig:
    """Configuration optimisée pour NGSpice en environnement parallèle"""
    
    @staticmethod
    def get_optimized_env_vars(n_parallel_jobs: int = 1) -> dict:
        """
        Génère les variables d'environnement optimales pour NGSpice
        
        Args:
            n_parallel_jobs: Nombre de simulations parallèles prévues
            
        Returns:
            Dict des variables d'environnement optimisées
        """
        import os
        import multiprocessing
        
        total_cores = multiprocessing.cpu_count()
        
        # ✅ Calcul intelligent des threads par job
        if n_parallel_jobs > 1:
            # Mode parallèle: limiter threads par instance
            threads_per_job = max(1, total_cores // (n_parallel_jobs * 2))
        else:
            # Mode séquentiel: utiliser plus de threads
            threads_per_job = max(2, total_cores // 2)
        
        env_vars = {
            **os.environ,
            
            # === CONTRÔLE DES THREADS ===
            # OpenMP (utilisé par NGSpice pour certaines opérations)
            'OMP_NUM_THREADS': str(threads_per_job),
            
            # Bibliothèques BLAS/LAPACK (algèbre linéaire)
            'MKL_NUM_THREADS': '1',           # Intel MKL
            'OPENBLAS_NUM_THREADS': '1',      # OpenBLAS
            'BLIS_NUM_THREADS': '1',          # BLIS
            'VECLIB_MAXIMUM_THREADS': '1',    # macOS Accelerate
            
            # === OPTIMISATIONS NGSPICE ===
            'NGSPICE_PRECISION': '3',         # ✅ 3-4 décimales suffisantes pour RL
            'NGSPICE_INPUT_DIR': '/tmp',      # Répertoire temporaire rapide
            
            # === CONTRÔLE MÉMOIRE ===
            'MALLOC_TRIM_THRESHOLD_': '100000',  # Libération mémoire plus fréquente
            'MALLOC_MMAP_THRESHOLD_': '50000',   # Réduction fragmentation
        }
        
        return env_vars
    
    @staticmethod
    def get_ngspice_options(fast_mode: bool = True) -> List[str]:
        """
        Options NGSpice pour optimiser vitesse vs précision
        
        Args:
            fast_mode: True pour RL (rapide), False pour validation finale
        """
        if fast_mode:
            return [
                'set ngbehavior=hsa',      # ✅ High-Speed Accuracy mode
                'set abstol=1e-9',         # ✅ Tolérance absolue relaxée
                'set reltol=0.01',         # ✅ Tolérance relative 1%
                'set vntol=1e-4',          # ✅ Tolérance tension
                'set chgtol=1e-12',        # ✅ Tolérance charge
                'set gmin=1e-10',          # ✅ Conductance minimale
                'set method=gear',         # ✅ Méthode intégration rapide
            ]
        else:
            # Mode précision pour validation
            return [
                'set ngbehavior=ps',       # Precision/Stability
                'set abstol=1e-12',
                'set reltol=0.001',
                'set vntol=1e-6',
                'set method=trap',         # Trapezoidal (plus stable)
            ]


# === INTÉGRATION DANS SequentialPool ===

class SequentialPool:
    """Pool séquentiel optimisé pour NGSpice"""
    
    def __init__(self, pdk: PDKManager, config: SimulationConfig, 
                 fast_mode: bool = True, verbose: bool = False):
        self.pdk = pdk
        self.config = config
        self.fast_mode = fast_mode
        self.verbose = verbose
        
        # ✅ Configuration optimisée
        self.opt_config = OptimizedNGSpiceConfig()
        self.env_vars = self.opt_config.get_optimized_env_vars(n_parallel_jobs=1)
        self.ngspice_opts = self.opt_config.get_ngspice_options(fast_mode)
        
    def run_batch(self, spice_files: List[Path]) -> pd.DataFrame:
        """Exécute un batch de simulations avec config optimisée"""
        results = []
        
        for spice_file in spice_files:
            try:
                # ✅ Ajout des options NGSpice au fichier
                optimized_netlist = self._inject_options(spice_file)
                
                # ✅ Exécution avec env_vars optimisés
                result = subprocess.run(
                    ['ngspice', '-b', optimized_netlist],
                    env=self.env_vars,  # ✅ Variables d'environnement
                    capture_output=True,
                    text=True,
                    timeout=10  # ✅ Timeout de sécurité
                )
                
                metrics = self._parse_output(result.stdout)
                results.append(metrics)
                
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print(f"⚠️  Timeout: {spice_file.name}")
                results.append({'error': 'timeout'})
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ Erreur: {spice_file.name} - {e}")
                results.append({'error': str(e)})
        
        return pd.DataFrame(results)
    
    def _inject_options(self, spice_file: Path) -> Path:
        """Injecte les options NGSpice dans le netlist"""
        with open(spice_file, 'r') as f:
            content = f.read()
        
        # Insertion après la première ligne (titre)
        lines = content.split('\n')
        options_block = '\n'.join(self.ngspice_opts)
        
        optimized_content = f"{lines[0]}\n{options_block}\n" + '\n'.join(lines[1:])
        
        # ✅ Fichier temporaire optimisé
        tmp_file = spice_file.parent / f"opt_{spice_file.name}"
        with open(tmp_file, 'w') as f:
            f.write(optimized_content)
        
        return tmp_file


# === INTÉGRATION DANS ParallelPool ===

class ParallelPool:
    """Pool parallèle avec contrôle des ressources"""
    
    def __init__(self, pdk: PDKManager, config: SimulationConfig,
                 n_workers: int = 4, fast_mode: bool = True, verbose: bool = False):
        self.pdk = pdk
        self.config = config
        self.n_workers = n_workers
        self.fast_mode = fast_mode
        self.verbose = verbose
        
        # ✅ Config optimisée pour mode parallèle
        self.opt_config = OptimizedNGSpiceConfig()
        self.env_vars = self.opt_config.get_optimized_env_vars(n_parallel_jobs=n_workers)
        self.ngspice_opts = self.opt_config.get_ngspice_options(fast_mode)
        
    def run_batch(self, spice_files: List[Path]) -> pd.DataFrame:
        """Exécute batch en parallèle avec ressources contrôlées"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # ✅ Soumettre jobs avec env_vars
            futures = {
                executor.submit(
                    self._run_single_simulation, 
                    spice_file, 
                    self.env_vars,  # ✅ Passer env_vars
                    self.ngspice_opts
                ): spice_file 
                for spice_file in spice_files
            }
            
            for future in as_completed(futures):
                spice_file = futures[future]
                try:
                    result = future.result(timeout=15)
                    results.append(result)
                except Exception as e:
                    if self.verbose:
                        print(f"❌ {spice_file.name}: {e}")
                    results.append({'error': str(e)})
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _run_single_simulation(spice_file: Path, env_vars: dict, 
                               ngspice_opts: List[str]) -> dict:
        """
        Fonction statique pour exécution parallèle
        (nécessaire pour ProcessPoolExecutor)
        """
        # ✅ Injection options
        opt_file = ParallelPool._inject_options_static(spice_file, ngspice_opts)
        
        try:
            result = subprocess.run(
                ['ngspice', '-b', opt_file],
                env=env_vars,  # ✅ Utilise env_vars optimisées
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return ParallelPool._parse_output_static(result.stdout)
            
        finally:
            # ✅ Nettoyage fichier temporaire
            if opt_file.exists():
                opt_file.unlink()
    
    @staticmethod
    def _inject_options_static(spice_file: Path, options: List[str]) -> Path:
        """Version statique pour parallélisation"""
        with open(spice_file, 'r') as f:
            lines = f.readlines()
        
        options_block = '\n'.join(options) + '\n'
        optimized = [lines[0], options_block] + lines[1:]
        
        tmp_file = spice_file.parent / f"opt_{spice_file.name}"
        with open(tmp_file, 'w') as f:
            f.writelines(optimized)
        
        return tmp_file


# === EXEMPLE D'UTILISATION ===

def example_usage():
    """Exemple complet d'utilisation optimisée"""
    
    # 1. Configuration
    pdk = PDKManager("sky130")
    config = SimulationConfig(vdd=1.8, temp=27)
    
    # 2. Mode séquentiel (entraînement rapide)
    seq_pool = SequentialPool(
        pdk, config,
        fast_mode=True,  # ✅ Mode rapide pour RL
        verbose=True
    )
    
    # 3. Mode parallèle (validation batch)
    par_pool = ParallelPool(
        pdk, config,
        n_workers=4,
        fast_mode=True,
        verbose=True
    )
    
    # 4. Exécution
    spice_files = list(Path("./sims").glob("*.cir"))
    
    print("🏃 Séquentiel:")
    df_seq = seq_pool.run_batch(spice_files[:10])
    
    print("\n⚡ Parallèle:")
    df_par = par_pool.run_batch(spice_files)
    
    # 5. Mode précision (validation finale)
    final_pool = SequentialPool(
        pdk, config,
        fast_mode=False,  # ✅ Haute précision
        verbose=True
    )
    df_final = final_pool.run_batch([spice_files[0]])  # Meilleur design


if __name__ == '__main__':
    example_usage()
