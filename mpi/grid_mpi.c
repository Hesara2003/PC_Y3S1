#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int in_degree[MAX_NODES];
int incoming_edges[MAX_NODES][100];

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
    Node *city = &nodes[city_id];
    int visited[MAX_NODES];
    int parent_edge[MAX_NODES];
    int queue[MAX_NODES];

    while (city->load < city->demand) {
        float needed = city->demand - city->load;
        memset(visited, 0, sizeof(visited));
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
            float flow = needed;
            if (nodes[found_gen].supply < flow)
                flow = nodes[found_gen].supply;
            int curr = found_gen;
            while (curr != city_id) {
                int e_idx = parent_edge[curr];
                if (edges[e_idx].capacity < flow)
                    flow = edges[e_idx].capacity;
                curr = edges[e_idx].to;
            }

            curr = found_gen;
            while (curr != city_id) {
                int e_idx = parent_edge[curr];
                edges[e_idx].capacity -= flow;
                curr = edges[e_idx].to;
            }
            nodes[found_gen].supply -= flow;
            city->load += flow;
        } else {
            break;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *filename = "data/grid.txt";
    if (argc > 1) {
        filename = argv[1];
    }

    load_grid(filename);

    /* Each rank holds an independent full-graph copy. Divide each generator's
     * supply by the number of ranks so the aggregate across all ranks equals
     * the true total supply — preventing artificial over-serving. */
    for (int i = 0; i < numNodes; i++) {
        if (nodes[i].type == TYPE_GEN) {
            nodes[i].supply /= (float)size;
        }
    }

    /* Similarly, scale edge capacities so aggregate flow capacity matches
     * the single-process baseline. */
    for (int i = 0; i < numEdges; i++) {
        edges[i].capacity /= (float)size;
    }

    int total_cities = 0;
    int city_indices[MAX_NODES];
    for (int i = 0; i < numNodes; i++) {
        if (nodes[i].type == TYPE_CITY) {
            city_indices[total_cities++] = i;
        }
    }

    int cities_per_rank = total_cities / size;
    int remainder = total_cities % size;

    int start_idx = rank * cities_per_rank + (rank < remainder ? rank : remainder);
    int count = cities_per_rank + (rank < remainder ? 1 : 0);
    int end_idx = start_idx + count;

    double start_time = MPI_Wtime();

    float local_served = 0.0f;
    float local_demand = 0.0f;

    for (int i = start_idx; i < end_idx; i++) {
        balance_city(city_indices[i]);
    }

    double end_time = MPI_Wtime();
    double local_duration = end_time - start_time;
    double max_duration;

    MPI_Reduce(&local_duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    for (int i = start_idx; i < end_idx; i++) {
        int nodeId = city_indices[i];
        local_served += nodes[nodeId].load;
        local_demand += nodes[nodeId].demand;
    }

    float global_served = 0;
    float global_demand = 0;

    MPI_Reduce(&local_served, &global_served, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_demand, &global_demand, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nTotal Demand Served: %.2f / %.2f\n", global_served, global_demand);
        printf("Time: %f\n", max_duration);
    }

    MPI_Finalize();
    return 0;
}
