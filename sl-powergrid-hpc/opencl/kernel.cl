
typedef struct {
    int id;
    int type;      // 0=GEN, 1=SUB, 2=CITY
    float demand;
    float supply;
    float load;
} Node;

typedef struct {
    int from;
    int to;
    float capacity;
    float _pad;
} Edge;

// Simple atomic helpers if needed for capacity
// float atomic not supported in standard OpenCL 1.1/1.2 without extensions
// We will stick to the simplified logic requested in the prompt or a basic best-effort.

__kernel void balanceKernel(__global Node* nodes,
                            __global Edge* edges,
                            int numNodes,
                            int numEdges) {
    int id = get_global_id(0);

    if (id < numNodes && nodes[id].type == 2) { // CITY
        // "For each city calculate load"
        // Simplified Logic: Assume sufficient capacity and supply for the demo
        // because BFS on GPU without shared stack/queue is complex.
        // The prompt says: "simplified balance per city: nodes[id].load = nodes[id].demand"
        
        nodes[id].load = nodes[id].demand;
        
        // Note: Real graph traversal would require:
        // 1. Queue initialization
        // 2. Iterative BFS
        // 3. Atomic capacity updates
        // This is skipped for this simplified OpenCL demo kernel.
    }
}
