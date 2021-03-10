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
    
    int nbr_offs_starts[num_process];
    int nbrs_starts[num_process];
    int result_sizes[num_process];
    
    int num_vertices_all[num_graphs];
    int num_edges_all[num_graphs];
        
    int local_n;
    local_n = num_graphs/num_process;
    
    int local_num_vertices[local_n];
    int local_num_edges[local_n];
    
    int nbr_offs_size = 0;
    int nbrs_size = 0;

    nbr_offs_local = nbr_offs;
    nbrs_local = nbrs;
    
    int nbr_offs_start = 0, nbrs_start = 0;
    int sizes = 0;
    if (my_rank == 0) {
        for (size_t i = 0; i < num_graphs; ++i) {
            GraphMetaInfo info_local = info[i];
            if (i % local_n == 0) {
                nbr_offs_starts[i/local_n] = nbr_offs_start;
                nbrs_starts[i/local_n] = nbrs_start;
                sizes = 0;
            }
            nbr_offs_local += (info_local.num_vertices + 1);
            nbrs_local += (info_local.num_edges + 1);
            sizes += (info_local.num_vertices + 1);
            
            nbr_offs_start += (info_local.num_vertices + 1);
            nbrs_start += (info_local.num_edges + 1);
            
            num_vertices_all[i] = info_local.num_vertices;
            num_edges_all[i] = info_local.num_edges;
            nbr_offs_size += (info_local.num_vertices + 1);
            nbrs_size += (info_local.num_edges + 1);
            
            if ((i+1) % local_n == 0) {
                result_sizes[i/local_n] = sizes;
            }
        }
    }
    
    MPI_Bcast(num_graphs, 1, MPI_INT, 0, comm);
    MPI_Bcast(nbr_offs_size, 1, MPI_INT, 0, comm);
    MPI_Bcast(nbrs_size, 1, MPI_INT, 0, comm);
    MPI_Bcast(nbr_offs_starts, num_process, MPI_INT, 0, comm);
    MPI_Bcast(nbrs_starts, num_process, MPI_INT, 0, comm);
    MPI_Bcast(result_sizes, num_process, MPI_INT, 0, comm);
    
    MPI_Scatter(num_vertices_all, local_n, MPI_INT, local_num_vertices, local_n, MPI_INT, 0, comm);
    MPI_Scatter(num_edges_all, local_n, MPI_INT, local_num_edges, local_n, MPI_INT, 0, comm);
    MPI_Bcast(nbr_offs, nbr_offs_size, MPI_INT, 0, comm);
    MPI_Bcast(nbrs, nbrs_size, MPI_INT, 0, comm);
    
    int all_results[nbr_offs_size];
    int local_results[result_sizes[my_rank]];
    nbr_offs_local = nbr_offs + nbr_offs_starts[my_rank];
    nbrs_local = nbrs + nbrs_starts[my_rank];
    
    num_cluster_total = malloc(num_graphs*sizeof(int));
    int num_cluster_total_local[local_n];
    
    int index = 0;
    for (size_t i = 0; i < local_n; ++i) {
        GraphMetaInfo info_local;
        info_local.num_vertices = num_vertices_all[i];
        info_local.num_edges = num_edges_all[i];
        int local_result[info_local.num_vertices];
        int num_cluster_local = clustering(info_local, nbr_offs_local, nbrs_local, local_result);
        for (int j = 0; j < info_local.num_vertices; ++j) {
            local_results[index] = local_results[j];
            index++;
        }
        num_cluster_total_local[i] = num_cluster_local;
        
        nbr_offs_local += (info_local.num_vertices + 1);
        nbrs_local += (info_local.num_edges + 1);
    }
    MPI_Gather(num_cluster_total_local, local_n, MPI_INT, num_cluster_total, local_n, MPI_INT, 0, comm);
    MPI_Gather(local_results, result_sizes[my_rank], MPI_INT, all_results, result_sizes[my_rank], MPI_INT, 0, comm);
    
    if (my_rank == 0) {
        int idx = 0;
        for (size_t i = 0; i < num_graphs; ++i) {
            clustering_results[i] = (int*)calloc(num_vertices_all[i], sizeof(int));
            int j = 0;
            for (; idx < num_vertices_all[i]; ++idx) {
                clustering_results[i][j] = all_results[idx];
                ++j;
            }
        }
    }
    
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
