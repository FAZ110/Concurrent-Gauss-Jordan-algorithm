import sys
import subprocess
import importlib.util
import os

# --- 1. AUTOMATYCZNA INSTALACJA BIBLIOTEK ---
def install_and_import(package):
    if importlib.util.find_spec(package) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception:
            pass

install_and_import("numpy")
install_and_import("networkx")
install_and_import("matplotlib")

import numpy as np
import concurrent.futures
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class GaussJordanFineGrained:
    def __init__(self, matrix_size):
        self.N = matrix_size
        self.alphabet = []  # Zbiór wszystkich unikalnych nazw czynności (np. N_0_0, E_1_0_2)
        self.trace = []     # Ślad, reprezentuje wykonanie sekwencyjne algorytmu
        self.dependencies = defaultdict(set)    # Reprezentacja relacji zależności, lista sąsiedztwa grafu skierowanego
        self.foata_classes = []     # Zbiór klas foaty
        # Zasoby to krotki (wiersz, kolumna). Kolumna N oznacza wektor b.
        self.op_resources = {}

    def define_operations(self):
        """Definiuje operacje na poziomie pojedynczych komórek."""
        for k in range(self.N):
            # --- FAZA A: NORMALIZACJA ---
            # 1. Normalizacja reszty wiersza i wektora b
            cols_to_normalize = list(range(k + 1, self.N + 1))
            for j in cols_to_normalize:
                op_name = f"N_{k}_{j}"
                self.alphabet.append(op_name)
                self.trace.append(op_name)
                # Czyta pivot (k,k), pisze swoją komórkę (k,j)
                self.op_resources[op_name] = {'read': {(k, k)}, 'write': {(k, j)}}

            # 2. Normalizacja samego pivota (ustawienie na 1)
            op_pivot = f"N_{k}_{k}"
            self.alphabet.append(op_pivot)
            self.trace.append(op_pivot)
            self.op_resources[op_pivot] = {'read': {(k, k)}, 'write': {(k, k)}}

            # --- FAZA B: ELIMINACJA ---
            for i in range(self.N):
                if i == k: continue
                
                # 1. Aktualizacja reszty wiersza i wektora b
                cols_to_eliminate = list(range(k + 1, self.N + 1))
                for j in cols_to_eliminate:
                    op_e = f"E_{i}_{k}_{j}"
                    self.alphabet.append(op_e)
                    self.trace.append(op_e)
                    # Czyta mnożnik (i,k), czyta źródło (k,j), pisze cel (i,j)
                    self.op_resources[op_e] = {'read': {(i, k), (k, j)}, 'write': {(i, j)}}
                
                # 2. Wyzerowanie komórki w kolumnie pivota
                op_e_pivot = f"E_{i}_{k}_{k}"
                self.alphabet.append(op_e_pivot)
                self.trace.append(op_e_pivot)
                self.op_resources[op_e_pivot] = {'read': {(k, k)}, 'write': {(i, k)}}

    def build_dependency_graph(self):
        history = [] 
        for op_curr in self.trace:
            res_curr = self.op_resources[op_curr]
            for op_prev in reversed(history):
                res_prev = self.op_resources[op_prev]
                dependent = False
                if not res_prev['write'].isdisjoint(res_curr['read']): dependent = True
                elif not res_prev['read'].isdisjoint(res_curr['write']): dependent = True
                elif not res_prev['write'].isdisjoint(res_curr['write']): dependent = True
                
                if dependent:
                    self.dependencies[op_curr].add(op_prev)
            history.append(op_curr)

    def compute_foata_classes(self):
        op_depth = {}
        for op in self.trace:
            deps = self.dependencies[op]
            if not deps:
                op_depth[op] = 1
            else:
                max_prev_depth = 0
                for prev_op in deps:
                    if prev_op in op_depth:
                        max_prev_depth = max(max_prev_depth, op_depth[prev_op])
                op_depth[op] = max_prev_depth + 1
        
        max_d = max(op_depth.values()) if op_depth else 0
        self.foata_classes = [[] for _ in range(max_d)]
        for op, d in op_depth.items():
            self.foata_classes[d-1].append(op)
        for i in range(len(self.foata_classes)):
            self.foata_classes[i].sort()

    def visualize_graph(self, filename="graf_komorkowy.png"):
        print(f"\nGenerowanie grafu {filename} (to może chwilę potrwać)...")
        try:
            G = nx.DiGraph()
            for child, parents in self.dependencies.items():
                for parent in parents:
                    G.add_edge(parent, child)
            
            pos = {}
            for layer_idx, layer_nodes in enumerate(self.foata_classes):
                y = -layer_idx * 3 
                width = len(layer_nodes)
                for i, node in enumerate(layer_nodes):
                    x = (i - width / 2.0) * 2
                    pos[node] = (x, y)

            plt.figure(figsize=(20, 15)) 
            
            colors = []
            for n in G.nodes():
                if n.startswith("N"): colors.append('#ffcc00')
                else: colors.append('#add8e6')

            nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300, 
                    font_size=5, arrowsize=8, edge_color='gray', width=0.3)
            
            plt.title(f"Graf Drobnoziarnisty (Cell-Level) - {len(self.alphabet)} operacji")
            plt.savefig(filename, format="PNG", dpi=150, bbox_inches='tight')
            plt.close()
            print(f">>> SUKCES: Zapisano graf: {filename}")
        except Exception as e:
            print(f"Błąd wizualizacji: {e}")

    # --- WYKONANIE (WORKERS) ---
    def execute_normalize_cell(self, matrix, vector, k, j):
        pivot = matrix[k][k]
        if j == self.N: vector[k] = vector[k] / pivot
        elif j == k:    matrix[k][k] = 1.0
        else:           matrix[k][j] = matrix[k][j] / pivot

    def execute_eliminate_cell(self, matrix, vector, i, k, j):
        multiplier = matrix[i][k]
        source_val = vector[k] if j == self.N else matrix[k][j]
        
        if j == self.N: vector[i] = vector[i] - multiplier * source_val
        elif j == k:    matrix[i][k] = 0.0
        else:           matrix[i][j] = matrix[i][j] - multiplier * source_val

    def solve(self, A, b):
        print("\n--- Rozpoczynam obliczenia drobnoziarniste ---")
        matrix = np.array(A, dtype=float)
        vector = np.array(b, dtype=float)
        
        # Więcej wątków, bo zadania są bardzo małe
        max_threads = max(4, self.N * self.N)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            for cls in self.foata_classes:
                futures = []
                for op in cls:
                    parts = op.split("_")
                    type_op = parts[0]
                    if type_op == "N":
                        k, j = int(parts[1]), int(parts[2])
                        futures.append(executor.submit(self.execute_normalize_cell, matrix, vector, k, j))
                    elif type_op == "E":
                        i, k, j = int(parts[1]), int(parts[2]), int(parts[3])
                        futures.append(executor.submit(self.execute_eliminate_cell, matrix, vector, i, k, j))
                
                for f in concurrent.futures.as_completed(futures):
                    f.result()
        return matrix, vector

# --- FUNKCJE POMOCNICZE (I/O) ---
def load_data_from_file(filename):
    try:
        with open(filename, 'r') as f:
            tokens = f.read().split()
            if not tokens: raise ValueError("Pusty plik")
            N = int(tokens[0])
            data = [float(x) for x in tokens[1:]]
            if len(data) < N*N + N: raise ValueError("Za mało danych")
            A = np.array(data[:N*N]).reshape(N, N)
            b = np.array(data[N*N : N*N+N])
            return N, A, b
    except Exception as e:
        print(f"Błąd wczytywania: {e}")
        sys.exit(1)

def verify_against_solution_file(filename, calc_vec, tolerance=1e-4):
    if not os.path.exists(filename): return
    try:
        with open(filename, 'r') as f:
            tokens = f.read().split()
        if not tokens: return
        N_sol = int(tokens[0])
        expected_vec = np.array([float(x) for x in tokens[-N_sol:]])
        if np.allclose(calc_vec, expected_vec, atol=tolerance):
            print("\n[OK] Wynik zgodny z plikiem solution.txt")
        else:
            diff = np.max(np.abs(calc_vec - expected_vec))
            print(f"\n[!] Wynik różni się od solution.txt. Max różnica: {diff}")
    except Exception: pass

def save_results_to_file(filename, matrix, vector):
    try:
        with open(filename, 'w') as f:
            N = len(vector)
            f.write(f"{N}\n")
            for row in matrix:
                f.write(" ".join(str(x) for x in row) + "\n")
            f.write(" ".join(str(x) for x in vector) + "\n")
        print(f"\n>>> Zapisano wynik do pliku: {filename} <<<")
    except Exception as e: print(f"Błąd zapisu: {e}")

# --- MAIN ---
def main():
    input_filename = 'input_data.txt'
    solution_filename = 'solution.txt'
    output_filename = 'my_result.txt'
    graph_filename = 'graf_komorkowy.png'

    print(f"--- Wczytywanie danych z {input_filename} ---")
    N, A, b = load_data_from_file(input_filename)
    print(f"Wczytano macierz {N}x{N}")

    solver = GaussJordanFineGrained(N)
    
    print("Definiowanie operacji (poziom komórkowy)...")
    solver.define_operations()
    
    print("Budowanie grafu zależności...")
    solver.build_dependency_graph()
    
    print("Wyznaczanie klas Foaty...")
    solver.compute_foata_classes()
    
    print(f"Liczba operacji elementarnych: {len(solver.alphabet)}")
    print(f"Liczba klas Foaty: {len(solver.foata_classes)}")
    
    
    # Generuj graf (tylko jeśli N nie jest zbyt duże, żeby nie zabić pamięci/czasu)
    if N <= 5:
        for foata_class in solver.foata_classes:
            print("-"*70)
            print(foata_class)
        solver.visualize_graph(graph_filename)
    else:
        print(f"\n[INFO] Pomijam generowanie grafu PNG, ponieważ N={N} jest duże.")
        print("       Graf drobnoziarnisty byłby nieczytelny.")

    res_A, res_x = solver.solve(A, b)
    
    verify_against_solution_file(solution_filename, res_x)
    save_results_to_file(output_filename, res_A, res_x)

if __name__ == "__main__":
    main()