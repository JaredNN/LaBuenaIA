# No.Control 22170741
# Noriega Nevarez Jared Emmanuel
# 24-Puzzle Solver usando IDA* con heurísticas de Manhattan y Conflicto Lineal

#15,500,000,000,000,000,000,000,000

import time
import random

# Tablero 5x5
GOAL_STATE = tuple(list(range(1, 25)) + [0])
N = 5

def get_inversions(state):
    inversions = 0
    state_list = [x for x in state if x != 0]
    for i in range(len(state_list)):
        for j in range(i + 1, len(state_list)):
            if state_list[i] > state_list[j]:
                inversions += 1
    return inversions

def is_solvable(state):
    """
    Para un tablero de NxN donde N es impar (como 5x5), 
    el puzzle es resoluble si el número de inversiones es par.
    """
    return get_inversions(state) % 2 == 0

def manhattan_distance(state):
    distance = 0
    for i in range(N * N):
        val = state[i]
        if val != 0:
            target_x = (val - 1) % N
            target_y = (val - 1) // N
            current_x = i % N
            current_y = i // N
            distance += abs(current_x - target_x) + abs(current_y - target_y)
    return distance

def linear_conflict(state):
    distance = manhattan_distance(state)
    conflicts = 0
    
    # conflictos en filas
    for row in range(N):
        for col1 in range(N):
            val1 = state[row * N + col1]
            if val1 != 0 and (val1 - 1) // N == row:
                for col2 in range(col1 + 1, N):
                    val2 = state[row * N + col2]
                    if val2 != 0 and (val2 - 1) // N == row:
                        if val1 > val2:
                            conflicts += 1
                            
    #  conflictos en columnas
    for col in range(N):
        for row1 in range(N):
            val1 = state[row1 * N + col]
            if val1 != 0 and (val1 - 1) % N == col:
                for row2 in range(row1 + 1, N):
                    val2 = state[row2 * N + col]
                    if val2 != 0 and (val2 - 1) % N == col:
                        if val1 > val2:
                            conflicts += 1
                            
    return distance + 2 * conflicts

def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    x, y = idx % N, idx // N
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # i d a b
    
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < N and 0 <= ny < N:
            n_idx = ny * N + nx
            new_state = list(state)
            new_state[idx], new_state[n_idx] = new_state[n_idx], new_state[idx]
            neighbors.append(tuple(new_state))
    return neighbors

def ida_star(start_state, heuristic_func):
    limit = heuristic_func(start_state)
    path = [start_state]
    nodes_expanded = [0]
    
    def search(path, g, limit):
        current = path[-1]
        f = g + heuristic_func(current)
        if f > limit:
            return f
        if current == GOAL_STATE:
            return "FOUND"
        
        min_cost = float('inf')
        nodes_expanded[0] += 1
        
        for neighbor in get_neighbors(current):
            if neighbor not in path:
                path.append(neighbor)
                res = search(path, g + 1, limit)
                if res == "FOUND":
                    return "FOUND"
                if res < min_cost:
                    min_cost = res
                path.pop()
        return min_cost

    start_time = time.time()
    while True:
        res = search(path, 0, limit)
        if res == "FOUND":
            end_time = time.time()
            return path, nodes_expanded[0], end_time - start_time
        if res == float('inf'):
            return None, nodes_expanded[0], time.time() - start_time
        limit = res

# Menú

def print_board(state):
    print("\n+" + "----+" * N)
    for i in range(N):
        row = state[i*N:(i+1)*N]
        row_str = "|"
        for val in row:
            if val == 0:
                row_str += "    |"
            else:
                row_str += f" {val:02d} |"
        print(row_str)
        print("+" + "----+" * N)
    print()

def input_manual_board():
    print("\n--- Configuración Manual ---")
    print("Ingresa los 25 números del tablero separados por espacios.")
    print("Usa los números del 0 al 24 (donde 0 es el espacio vacío).")
    print("Ejemplo de tablero resuelto: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 0")
    
    while True:
        try:
            user_input = input("\nIngresa los números: ")
            numbers = [int(x) for x in user_input.split()]
            
            if len(numbers) != 25:
                print(f"Error: Has ingresado {len(numbers)} números. Deben ser exactamente 25.")
                continue
            if set(numbers) != set(range(25)):
                print("Error: Los números deben ser únicos y estar en el rango de 0 a 24.")
                continue
                
            state = tuple(numbers)
            if not is_solvable(state):
                print("Error: Esta configuración NO es resoluble (paridad de inversiones incorrecta). Intenta otra.")
                continue
                
            return state
        except ValueError:
            print("Error: Formato inválido. Asegúrate de ingresar solo números separados por espacios.")

def generate_random_board():
    print("\n--- Configuración Aleatoria ---")
    print("Para garantizar que la prueba se resuelva en un tiempo razonable,")
    print("se generará realizando movimientos aleatorios hacia atrás desde la meta.")
    
    while True:
        try:
            dificultad = int(input("Ingresa la dificultad (número de movimientos a desordenar, ej. 20): "))
            if dificultad < 1:
                print("Por favor, ingresa un número mayor a 0.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Ingresa un número entero.")
            
    state = list(GOAL_STATE)
    for _ in range(dificultad):
        state = random.choice(get_neighbors(tuple(state)))
    return tuple(state)

def run_solver(start_state, heuristic_func, heuristic_name):
    print(f"\nEjecutando IDA* con la heurística: {heuristic_name}...")
    path, nodes, exec_time = ida_star(start_state, heuristic_func)
    
    if path:
        print(f"Solución encontrada!!")
        print(f"- Longitud de la solución (movimientos): {len(path) - 1}")
        print(f"- Nodos expandidos: {nodes}")
        print(f"- Tiempo de ejecución: {exec_time:.4f} segundos")
    else:
        print("No se pudo encontrar una solución.")

def main():
    while True:
        print("\n" + "="*40)
        print("   RESOLVEDOR DEL 24-PUZZLE (IDA*)")
        print("="*40)
        print("1. Ingresar configuración inicial manual")
        print("2. Generar configuración aleatoria resoluble")
        print("3. Salir")
        
        opcion = input("\nElige una opción (1-3): ")
        
        if opcion == '1':
            start_state = input_manual_board()
        elif opcion == '2':
            start_state = generate_random_board()
        elif opcion == '3':
            print("Ya terminaste, al rato.")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
            continue
            
        print("\nTablero Inicial Seleccionado:")
        print_board(start_state)
        
        # Submenú de heurísticas
        while True:
            print("\nOpciones de Heurística:")
            print("1. Distancia de Manhattan")
            print("2. Conflicto Lineal")
            print("3. Comparar ambas heurísticas")
            print("4. Volver al menú principal")
            
            h_opcion = input("\nElige una heurística (1-4): ")
            
            if h_opcion == '1':
                run_solver(start_state, manhattan_distance, "Distancia de Manhattan")
                break
            elif h_opcion == '2':
                run_solver(start_state, linear_conflict, "Conflicto Lineal")
                break
            elif h_opcion == '3':
                run_solver(start_state, manhattan_distance, "Distancia de Manhattan")
                run_solver(start_state, linear_conflict, "Conflicto Lineal")
                break
            elif h_opcion == '4':
                break # Sale del bucle de heurística, vuelve al menú principal
            else:
                print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()