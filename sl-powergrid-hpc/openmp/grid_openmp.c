#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_NODES 1000
#define MAX_EDGES 10000
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

// Adjacency list (shared, read-only during BFS navigation)
int in_degree[MAX_NODES];
int incoming_edges[MAX_NODES][100];

void load_grid(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    printf("Error opening file\n");
    exit(1);
  }
  // ... same loading logic as serial ...
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
      int id;
      char type_str[20];
      float val1, val2;
      if (sscanf(line, "%d %s %f %f", &id, type_str, &val1, &val2) == 4) {
        nodes[id].id = id;
        nodes[id].demand = val1;
        nodes[id].supply = val2;
        nodes[id].load = 0;
        if (strcmp(type_str, "GENERATOR") == 0)
          nodes[id].type = TYPE_GEN;
        else if (strcmp(type_str, "SUBSTATION") == 0)
          nodes[id].type = TYPE_SUB;
        else
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
        if (in_degree[v] < 100)
          incoming_edges[v][in_degree[v]++] = numEdges;
        numEdges++;
      }
    }
  }
  fclose(f);
}

void balance_city(int city_id) {
  // Thread-private structures
  int visited[MAX_NODES];
  int parent_edge[MAX_NODES];
  int queue[MAX_NODES];

  Node *city = &nodes[city_id];

  while (1) {
    float current_load;
    // Read load atomically to check if done?
    // Logic: specific thread handles specific city. No other thread updates
    // THIS city's load. So standard read is fine.
    if (city->load >= city->demand)
      break;

    float needed = city->demand - city->load;

    // BFS
    memset(visited, 0, sizeof(visited));
    int q_front = 0, q_rear = 0;

    queue[q_rear++] = city_id;
    visited[city_id] = 1;

    int found_gen = -1;

    while (q_front < q_rear) {
      int curr = queue[q_front++];

      // Check if GEN. Read supply atomically?
      // We need a snapshot. Just read. We will atomic update later.
      if (nodes[curr].type == TYPE_GEN && nodes[curr].supply > 0) {
        found_gen = curr;
        break;
      }

      for (int i = 0; i < in_degree[curr]; i++) {
        int edge_idx = incoming_edges[curr][i];
        int neighbor = edges[edge_idx].from;

        // Read capacity. Note: capacity changes dynamically.
        // We use current snapshot.
        if (!visited[neighbor] && edges[edge_idx].capacity > 0) {
          visited[neighbor] = 1;
          parent_edge[neighbor] = edge_idx;
          queue[q_rear++] = neighbor;
        }
      }
    }

    if (found_gen != -1) {
      float flow = needed;

      // Limit by supply (snapshot)
      float s = nodes[found_gen].supply;
      if (s < flow)
        flow = s;

      // Limit by edge capacities (snapshot)
      int curr = found_gen;
      while (curr != city_id) {
        int e_idx = parent_edge[curr];
        float c = edges[e_idx].capacity;
        if (c < flow)
          flow = c;
        curr = edges[e_idx].to;
      }

      if (flow < 0.001f)
        break; // Avoid infinite minimal updates

      // Apply updates
      // We must traverse again and ATOMICALLY subtract.
      // Be careful: capacities might have changed since check.
      // We proceed with the calculated 'flow'. In a real strict system, we'd
      // need reservations. Here, we follow the "atomic subtract" instruction.

      curr = found_gen;
      while (curr != city_id) {
        int e_idx = parent_edge[curr];
#pragma omp atomic
        edges[e_idx].capacity -= flow;
        curr = edges[e_idx].to;
      }

#pragma omp atomic
      nodes[found_gen].supply -= flow;

      // Update local city load (private to this thread effectively)
      city->load += flow;

    } else {
      break;
    }
  }
}

int main() {
  load_grid("data/grid.txt");

  double start = omp_get_wtime();

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < numNodes; i++) {
    if (nodes[i].type == TYPE_CITY) {
      balance_city(i);
    }
  }

  double end = omp_get_wtime();

  float total_served = 0;
  float total_demand = 0;
  for (int i = 0; i < numNodes; i++) {
    if (nodes[i].type == TYPE_CITY) {
      total_served += nodes[i].load;
      total_demand += nodes[i].demand;
      printf("Node %d Load: %.2f / %.2f\n", nodes[i].id, nodes[i].load,
             nodes[i].demand);
    }
  }

  printf("\nTotal Demand Served: %.2f / %.2f\n", total_served, total_demand);
  printf("Time: %f\n", end - start);

  return 0;
}
