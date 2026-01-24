#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_NODES 1000
#define MAX_EDGES 10000
#define TYPE_GEN 0
#define TYPE_SUB 1
#define TYPE_CITY 2

// Must match kernel struct layout!
typedef struct {
  int id;
  int type; // 0=GEN, 1=SUB, 2=CITY
  float demand;
  float supply;
  float load;
} Node;

typedef struct {
  int from;
  int to;
  float capacity;
  // padding might be needed for alignment in real OpenCL,
  // but for this simple struct (2 ints, 1 float = 12 bytes), packed vs aligned
  // issues might occur. relying on luck or standard packing for 101.
  float _pad;
} Edge;

Node nodes[MAX_NODES];
Edge edges[MAX_EDGES]; // Using struct with pad
int numNodes = 0;
int numEdges = 0;

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
        edges[numEdges]._pad = 0;
        numEdges++;
      }
    }
  }
  fclose(f);
}

const char *getErrorString(cl_int error) {
  switch (error) {
  case 0:
    return "CL_SUCCESS";
  case -11:
    return "CL_BUILD_PROGRAM_FAILURE";
  // Add more if needed
  default:
    return "Unknown OpenCL Error";
  }
}

int main() {
  load_grid("data/grid.txt");

  // OpenCL Setup
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err < 0) {
    printf("Error: clGetPlatformIDs %s\n", getErrorString(err));
    return 1;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err == CL_DEVICE_NOT_FOUND) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  }
  if (err < 0) {
    printf("Error: clGetDeviceIDs %s\n", getErrorString(err));
    return 1;
  }

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err < 0) {
    printf("Error: clCreateContext %s\n", getErrorString(err));
    return 1;
  }

  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err < 0) {
    printf("Error: clCreateCommandQueue %s\n", getErrorString(err));
    return 1;
  }

  // Read Kernel Source
  FILE *fp = fopen("opencl/kernel.cl",
                   "rb"); // Open in binary mode to avoid CRLF issues
  if (!fp) {
    printf("Error: Cannot open opencl/kernel.cl\n");
    return 1;
  }
  fseek(fp, 0, SEEK_END);
  size_t src_size = ftell(fp);
  rewind(fp);
  char *source_str = (char *)malloc(src_size + 1);
  size_t val = fread(source_str, 1, src_size, fp);
  source_str[val] = '\0';
  fclose(fp);

  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&src_size, &err);
  if (err < 0) {
    printf("Error: clCreateProgramWithSource %s\n", getErrorString(err));
    return 1;
  }

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err < 0) {
    printf("Error: clBuildProgram %s\n", getErrorString(err));
    char buffer[10240];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer),
                          buffer, NULL);
    printf("Build Log: %s\n", buffer);
    return 1;
  }

  kernel = clCreateKernel(program, "balanceKernel", &err);
  if (err < 0) {
    printf("Error: clCreateKernel %s\n", getErrorString(err));
    return 1;
  }

  // Create Buffers
  cl_mem nodes_buf =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(Node) * MAX_NODES, nodes, &err);
  cl_mem edges_buf =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(Edge) * MAX_EDGES, edges, &err);

  // Set Args
  int n_nodes = numNodes; // pass as value
  int n_edges = numEdges;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &nodes_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &edges_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(int), &n_nodes);
  err |= clSetKernelArg(kernel, 3, sizeof(int), &n_edges);

  // Execute
  clock_t start = clock();
  size_t global_work_size = MAX_NODES; // One thread per potential node
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL,
                               0, NULL, NULL);
  if (err < 0) {
    printf("Error: clEnqueueNDRangeKernel %s\n", getErrorString(err));
    return 1;
  }

  clFinish(queue);

  // Read back
  err = clEnqueueReadBuffer(queue, nodes_buf, CL_TRUE, 0,
                            sizeof(Node) * MAX_NODES, nodes, 0, NULL, NULL);

  clock_t end = clock();

  // Results
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
  printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);

  clReleaseMemObject(nodes_buf);
  clReleaseMemObject(edges_buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(source_str);

  return 0;
}
