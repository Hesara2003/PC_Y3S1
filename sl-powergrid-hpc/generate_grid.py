import random
import sys

def generate_grid(filename, num_nodes, num_edges):
    with open(filename, 'w') as f:
        f.write("# nodes\n")
        # Ensure at least 1 generator and 1 city
        nodes = []
        # Node 0: Generator
        f.write(f"0 GENERATOR 0 {num_nodes * 100}\n")
        nodes.append(0)
        
        # Intermediate nodes (Substations)
        # Last 20% nodes are Cities
        num_cities = max(1, int(num_nodes * 0.2))
        num_subs = num_nodes - 1 - num_cities
        
        current_id = 1
        for _ in range(num_subs):
            f.write(f"{current_id} SUBSTATION 0 0\n")
            nodes.append(current_id)
            current_id += 1
            
        for _ in range(num_cities):
            demand = random.randint(50, 200)
            f.write(f"{current_id} CITY {demand} 0\n")
            nodes.append(current_id)
            current_id += 1
            
        f.write("\n# edges\n")
        
        # Create a spanning tree to ensure connectivity
        edges_count = 0
        for i in range(1, num_nodes):
            parent = random.randint(0, i-1)
            cap = random.randint(500, 1000)
            f.write(f"{parent} {i} {cap}\n")
            edges_count += 1
            
        # Add random edges
        while edges_count < num_edges:
            u = random.randint(0, num_nodes-1)
            v = random.randint(0, num_nodes-1)
            if u != v:
                cap = random.randint(500, 1000)
                f.write(f"{u} {v} {cap}\n")
                edges_count += 1

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: generate_grid.py <filename> <nodes> <edges>")
        sys.exit(1)
        
    filename = sys.argv[1]
    n = int(sys.argv[2])
    e = int(sys.argv[3])
    generate_grid(filename, n, e)
    print(f"Generated {filename} with {n} nodes and {e} edges")
