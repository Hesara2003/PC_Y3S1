#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_NODES 100000
#define MAX_EDGES 300000
#define TYPE_GEN 0
#define TYPE_SUB 1
#define TYPE_CITY 2

typedef struct {
  int id;
  int type;
  float demand;
  float supply;
  float load;
} Node;

typedef struct {
  int from;
  int to;
  float capacity;
} Edge;

Node nodes[MAX_NODES];
Edge edges[MAX_EDGES];
int numNodes = 0;
int numEdges = 0;

// Adjacency list for reverse graph traversal (to find path from Gen -> City
// starting from City) actually, standard BFS from Gen to City is fine, but
// prompt says "For each city... find path". Efficient way: Backward BFS from
// City to find a Generator. So we need 'incoming' edges for each node.
int in_degree[MAX_NODES];
int incoming_edges[MAX_NODES][100]; // store indices of edges incoming to a node

void load_grid(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    printf("Error opening file\n");
    exit(1);
  }

  char line[256];
  int reading_nodes = 0;
  int reading_edges = 0;

  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "# nodes", 7) == 0) {
      reading_nodes = 1;
      reading_edges = 0;
      continue;
    }
    if (strncmp(line, "# edges", 7) == 0) {
      reading_nodes = 0;
      reading_edges = 1;
      continue;
    }
    if (strlen(line) < 2)
      continue;

    if (reading_nodes) {
      int id, type;
      float val1, val2; // demand/supply depending on type
      // Based on example: 0 GENERATOR 0 500 -> id TYPE demand supply
      // 2 CITY 120 0 -> id TYPE demand supply
      // Wait, example: "0 GENERATOR 0 500"
      // "2 CITY 120 0"
      // So format is: ID TYPE_STR DEMAND SUPPLY
      char type_str[20];
      if (sscanf(line, "%d %s %f %f", &id, type_str, &val1, &val2) == 4) {
        nodes[id].id = id;
        nodes[id].demand = val1;
        nodes[id].supply = val2;
        nodes[id].load = 0;

        if (strcmp(type_str, "GENERATOR") == 0)
          nodes[id].type = TYPE_GEN;
        else if (strcmp(type_str, "SUBSTATION") == 0)
          nodes[id].type = TYPE_SUB;
        else if (strcmp(type_str, "CITY") == 0)
          nodes[id].type = TYPE_CITY;

        if (id >= numNodes)
          numNodes = id + 1;
      }
    } else if (reading_edges) {
      int u, v;
      float cap;
      if (sscanf(line, "%d %d %f", &u, &v, &cap) == 3) {
        edges[numEdges].from = u;
        edges[numEdges].to = v;
        edges[numEdges].capacity = cap;

        // Add to adjacency
        if (in_degree[v] < 100) {
          incoming_edges[v][in_degree[v]++] = numEdges;
        }
        numEdges++;
      }
    }
  }
  fclose(f);
}

// Simple queue for BFS
int queue[MAX_NODES];
int parent_edge[MAX_NODES]; // store edge index used to reach this node
int visited[MAX_NODES];

void balance_city(int city_id) {
  Node *city = &nodes[city_id];

  // While city needs power
  while (city->load < city->demand) {
    float needed = city->demand - city->load;

    // BFS to find a generator with supply > 0
    // We search BACKWARDS from City to Generator
    int q_front = 0, q_rear = 0;

    for (int i = 0; i < numNodes; i++) {
      visited[i] = 0;
      parent_edge[i] = -1;
    }

    queue[q_rear++] = city_id;
    visited[city_id] = 1;

    int found_gen = -1;

    while (q_front < q_rear) {
      int curr = queue[q_front++];

      if (nodes[curr].type == TYPE_GEN && nodes[curr].supply > 0) {
        found_gen = curr;
        break;
      }

      // Check incoming edges
      for (int i = 0; i < in_degree[curr]; i++) {
        int edge_idx = incoming_edges[curr][i];
        // Edge is from -> curr
        // We go backwards to 'from'
        int neighbor = edges[edge_idx].from;

        if (!visited[neighbor] && edges[edge_idx].capacity > 0) {
          visited[neighbor] = 1;
          parent_edge[neighbor] =
              edge_idx; // We came from 'neighbor' to 'curr' via this edge
          queue[q_rear++] = neighbor;
        }
      }
    }

    if (found_gen != -1) {
      // Trace path forward from Gen to City
      // But parent_edge stores how we stepped BACK to 'neighbor'.
      // parent_edge[neighbor] stores the edge connecting neighbor -> curr
      // (where curr is closer to city) Wait. When I went curr -> neighbor
      // (backwards), I verified edge matches neighbor->curr. So
      // parent_edge[neighbor] stores edge index for neighbor ->
      // some_node_closer_to_city NO. Let's re-verify. Start at City. Push City
      // to Q. Use edge incoming to City (U->City). Visit U. parent_edge[U] =
      // edge (U->City). Use edge incoming to U (V->U). Visit V. parent_edge[V]
      // = edge (V->U). Found Gen G. parent_edge[G] = edge (G->Sub).

      // So to trace path: start at G, use parent_edge[G] to find next node.

      float flow = needed;
      if (nodes[found_gen].supply < flow)
        flow = nodes[found_gen].supply;

      // First pass: find bottleneck capacity
      int curr = found_gen;
      while (curr != city_id) {
        int e_idx = parent_edge[curr];
        if (edges[e_idx].capacity < flow)
          flow = edges[e_idx].capacity;
        curr = edges[e_idx].to;
      }

      // Second pass: update values
      curr = found_gen;
      while (curr != city_id) {
        int e_idx = parent_edge[curr];
        edges[e_idx].capacity -= flow;
        curr = edges[e_idx].to;
      }

      nodes[found_gen].supply -= flow;
      city->load += flow;

    } else {
      // No path found
      break;
    }
  }
}

// --- Unit Testing Helpers ---

void clear_grid() {
  numNodes = 0;
  numEdges = 0;
  for (int i = 0; i < MAX_NODES; i++) {
    in_degree[i] = 0;
    nodes[i].id = 0;
    nodes[i].type = 0;
    nodes[i].demand = 0;
    nodes[i].supply = 0;
    nodes[i].load = 0;
  }
}

void add_node(int id, int type, float demand, float supply) {
  nodes[id].id = id;
  nodes[id].type = type;
  nodes[id].demand = demand;
  nodes[id].supply = supply;
  nodes[id].load = 0;
  if (id >= numNodes)
    numNodes = id + 1;
}

void add_edge(int from, int to, float capacity) {
  edges[numEdges].from = from;
  edges[numEdges].to = to;
  edges[numEdges].capacity = capacity;
  if (in_degree[to] < 100) {
    incoming_edges[to][in_degree[to]++] = numEdges;
  }
  numEdges++;
}

void run_unit_tests() {
  printf("Running Unit Tests...\n");

  // Test 1: 1 Generator -> 1 City (Supply: 100, Demand: 50 -> Expected Served:
  // 50)
  clear_grid();
  add_node(0, TYPE_GEN, 0, 100);
  add_node(1, TYPE_CITY, 50, 0);
  add_edge(0, 1, 100);

  balance_city(1);

  if (nodes[1].load == 50.0f) {
    printf("[PASS] Graph Load (1 Gen -> 1 City)\n");
  } else {
    printf("[FAIL] Graph Load: Expected 50.0, got %.2f\n", nodes[1].load);
  }

  // Test 2: Capacity Constraints (Gen: 100, Edge Cap: 30, Demand: 50 ->
  // Expected Served: 30)
  clear_grid();
  add_node(0, TYPE_GEN, 0, 100);
  add_node(1, TYPE_CITY, 50, 0);
  add_edge(0, 1, 30); // Bottleneck

  balance_city(1);

  if (nodes[1].load == 30.0f) {
    printf("[PASS] Capacity Enforcement\n");
  } else {
    printf("[FAIL] Capacity Enforcement: Expected 30.0, got %.2f\n",
           nodes[1].load);
  }

  // Test 3: Multiple Cities (Shared Resource)
  // Gen (100) -> Sub -> City1 (40)
  //                  -> City2 (40)
  // Both should be fully served.
  clear_grid();
  add_node(0, TYPE_GEN, 0, 100);
  add_node(1, TYPE_SUB, 0, 0);
  add_node(2, TYPE_CITY, 40, 0);
  add_node(3, TYPE_CITY, 40, 0);

  add_edge(0, 1, 100);
  add_edge(1, 2, 100);
  add_edge(1, 3, 100);

  balance_city(2);
  balance_city(3);

  if (nodes[2].load == 40.0f && nodes[3].load == 40.0f) {
    printf("[PASS] Flow Calculation (Multiple Cities)\n");
  } else {
    printf("[FAIL] Flow Calculation: Expected 40/40, got %.2f/%.2f\n",
           nodes[2].load, nodes[3].load);
  }
}

// --- Generator Helper ---
void generate_random_grid(const char *filename, int n, int e) {
  FILE *f = fopen(filename, "w");
  if (!f) {
    printf("Error creating file %s\n", filename);
    return;
  }

  // Nodes
  fprintf(f, "# nodes\n");
  // 0 is Generator
  // 1..n-1 are mix of Substation/City
  // Lets make 20% cities
  srand((unsigned int)time(NULL));

  fprintf(f, "0 GENERATOR 0 %d\n", n * 100);

  int num_cities = n / 5;
  if (num_cities < 1)
    num_cities = 1;

  for (int i = 1; i < n; i++) {
    if (i < (n - num_cities)) {
      fprintf(f, "%d SUBSTATION 0 0\n", i);
    } else {
      int demand = (rand() % 151) + 50; // 50-200
      fprintf(f, "%d CITY %d 0\n", i, demand);
    }
  }

  // Edges
  fprintf(f, "\n# edges\n");
  // random tree + extras
  for (int i = 1; i < n; i++) {
    int parent = rand() % i;
    int cap = (rand() % 501) + 500; // 500-1000
    fprintf(f, "%d %d %d\n", parent, i, cap);
  }

  int current_edges = n - 1;
  while (current_edges < e) {
    int u = rand() % n;
    int v = rand() % n;
    if (u != v) {
      int cap = (rand() % 501) + 500;
      fprintf(f, "%d %d %d\n", u, v, cap);
      current_edges++;
    }
  }

  fclose(f);
  printf("Generated %s with %d nodes, %d edges\n", filename, n, e);
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--test") == 0) {
    run_unit_tests();
    return 0;
  }

  if (argc > 4 && strcmp(argv[1], "--generate") == 0) {
    generate_random_grid(argv[2], atoi(argv[3]), atoi(argv[4]));
    return 0;
  }

  const char *filename = "data/grid.txt";
  if (argc > 1 && strncmp(argv[1], "--", 2) != 0) {
    filename = argv[1];
  }

  load_grid(filename);

  clock_t start = clock();

  // Main loop
  for (int i = 0; i < numNodes; i++) {
    if (nodes[i].type == TYPE_CITY) {
      balance_city(i);
    }
  }

  clock_t end = clock();

  // Results
  float total_served = 0;
  float total_demand = 0;
  for (int i = 0; i < numNodes; i++) {
    if (nodes[i].type == TYPE_CITY) {
      total_served += nodes[i].load;
      total_demand += nodes[i].demand;
      if (numNodes < 50) {
        printf("Node %d Load: %.2f / %.2f\n", nodes[i].id, nodes[i].load,
               nodes[i].demand);
      }
    }
  }

  printf("\nTotal Demand Served: %.2f / %.2f\n", total_served, total_demand);
  printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

  return 0;
}
