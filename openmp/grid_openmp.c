#include <omp.h>
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
            if (sscanf(line, "%d %19s %f %f", &id, type_str, &val1, &val2) == 4) {
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
    // Thread-private structures - Allocate on HEAP to avoid Stack Overflow
    int *visited = (int *)malloc(MAX_NODES * sizeof(int));
    int *parent_edge = (int *)malloc(MAX_NODES * sizeof(int));
    int *queue = (int *)malloc(MAX_NODES * sizeof(int));

    if (!visited || !parent_edge || !queue) {
        if (visited)
            free(visited);
        if (parent_edge)
            free(parent_edge);
        if (queue)
            free(queue);
        return; // Allocation failure
    }

    Node *city = &nodes[city_id];

    while (1) {
        // Read load atomically?
        // Since only this thread writes to THIS city's load, dirty read is 'safe
        // enough' for loop condition. We will re-check strictly inside critical
        // section if needed (not really, load is private).
        if (city->load >= city->demand)
            break;

        float needed = city->demand - city->load;

        // BFS (Parallel Read Phase)
        // We search based on *snapshot* of state.
        memset(visited, 0, MAX_NODES * sizeof(int));
        int q_front = 0, q_rear = 0;

        queue[q_rear++] = city_id;
        visited[city_id] = 1;

        int found_gen = -1;

        while (q_front < q_rear) {
            int curr = queue[q_front++];

            if (nodes[curr].type == TYPE_GEN && nodes[curr].supply > 0) {
                found_gen = curr;
                break;
            }

            for (int i = 0; i < in_degree[curr]; i++) {
                int edge_idx = incoming_edges[curr][i];
                int neighbor = edges[edge_idx].from;

                if (!visited[neighbor] && edges[edge_idx].capacity > 0) {
                    visited[neighbor] = 1;
                    parent_edge[neighbor] = edge_idx;
                    queue[q_rear++] = neighbor;
                }
            }
        }

        if (found_gen != -1) {
            int success = 0;

// CRITICAL SECTION: Verify and Update
// We must re-check if the path found is still valid, because other threads
// modified the graph.
#pragma omp critical
            {
                // Re-calculate flow based on CURRENT STRICT state
                float flow = needed;

                // Check Generator
                float s = nodes[found_gen].supply;
                if (s < flow)
                    flow = s;

                if (flow > 0) {
                    // Check Path Capacity
                    int curr = found_gen;
                    int path_valid = 1;
                    while (curr != city_id) {
                        int e_idx = parent_edge[curr];
                        if (edges[e_idx].capacity < flow) {
                            flow = edges[e_idx].capacity;
                        }
                        if (flow <= 0.0001f) {
                            path_valid = 0;
                            break;
                        }
                        curr = edges[e_idx].to;
                    }

                    if (path_valid && flow > 0.0001f) {
                        // Apply Update (Atomic by virtue of Critical Section)
                        curr = found_gen;
                        while (curr != city_id) {
                            int e_idx = parent_edge[curr];
                            edges[e_idx].capacity -= flow;
                            curr = edges[e_idx].to;
                        }
                        nodes[found_gen].supply -= flow;
                        city->load += flow;
                        success = 1;
                    }
                }
            } // End Critical

            if (!success) {
                // If we found a path but it became invalid, we should try BFS again
                // or if flow was 0, break to avoid infinite loop
                // Simple heuristic: if we learned nothing new (no path), break.
                // If we found path but failed verify, loop again.
                continue;
            }

        } else {
            // No path found in BFS
            break;
        }
    }

    free(visited);
    free(parent_edge);
    free(queue);
}

int main(int argc, char **argv) {
    const char *filename = "data/grid.txt";
    if (argc > 1) {
        filename = argv[1];
    }
    printf("Loading grid from: %s\n", filename);
    load_grid(filename);

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
            printf("Node %d Load: %.2f / %.2f\n", nodes[i].id, nodes[i].load, nodes[i].demand);
        }
    }

    printf("\nTotal Demand Served: %.2f / %.2f\n", total_served, total_demand);
    printf("Time: %f\n", end - start);

    return 0;
}
