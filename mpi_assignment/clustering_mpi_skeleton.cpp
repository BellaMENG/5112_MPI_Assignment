#include "clustering.h"

#include "mpi.h"

#include <cassert>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  int num_process; // number of processors
  int my_rank;     // my global rank

  comm = MPI_COMM_WORLD;

  MPI_Comm_size(comm, &num_process);
  MPI_Comm_rank(comm, &my_rank);

  if (argc != 3) {
    std::cerr << "usage: ./clustering_sequential data_path result_path"
              << std::endl;

    return -1;
  }
  std::string dir(argv[1]);
  std::string result_path(argv[2]);

  int num_graphs;
  int *clustering_results = nullptr;
  int *num_cluster_total = nullptr;

  int *nbr_offs = nullptr, *nbrs = nullptr;
  int *nbr_offs_local = nullptr, *nbrs_local = nullptr;

  GraphMetaInfo *info = nullptr;

  // read graph info from files
  if (my_rank == 0) {
    num_graphs = read_files(dir, info, nbr_offs, nbrs);
  }
  auto start_clock = chrono::high_resolution_clock::now();

  // ADD THE CODE HERE

  MPI_Barrier(comm);
  auto end_clock = chrono::high_resolution_clock::now();

  // 1) print results to screen
  if (my_rank == 0) {
    for (size_t i = 0; i < num_graphs; i++) {
      printf("num cluster in graph %d : %d\n", i, num_cluster_total[i]);
    }
    fprintf(stderr, "Elapsed Time: %.9lf ms\n",
            chrono::duration_cast<chrono::nanoseconds>(end_clock - start_clock)
                    .count() /
                pow(10, 6));
  }

  // 2) write results to file
  if (my_rank == 0) {
    int *result_graph = clustering_results;
    for (int i = 0; i < num_graphs; i++) {
      GraphMetaInfo info_local = info[i];
      write_result_to_file(info_local, i, num_cluster_total[i], result_graph,
                           result_path);

      result_graph += info_local.num_vertices;
    }
  }

  MPI_Finalize();

  if (my_rank == 0) {
    free(num_cluster_total);
  }

  return 0;
}
