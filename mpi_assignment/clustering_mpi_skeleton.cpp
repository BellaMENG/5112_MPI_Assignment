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
    
    // have sizes of num_process
    int *nbr_offs_starts = nullptr;
    int *nbrs_starts = nullptr;
    int *result_sizes = nullptr;
    
    // have sizes of num_graphs
    int *num_vertices_all = nullptr;
    int *num_edges_all = nullptr;
        
    int local_n;
    // have sizes of local_n
    int *local_num_vertices = nullptr;
    int *local_num_edges = nullptr;
    
    int nbr_offs_size = 0;
    int nbrs_size = 0;
    int num_vertices_sum = 0;

    nbr_offs_local = nbr_offs;
    nbrs_local = nbrs;
    
    int nbr_offs_start = 0, nbrs_start = 0;
    int sizes = 0;
    
    if (my_rank == 0) {
        local_n = num_graphs/num_process;
        nbr_offs_starts = (int*)malloc(num_process * sizeof(int));
        nbrs_starts = (int*)malloc(num_process * sizeof(int));
        result_sizes = (int*)malloc(num_process * sizeof(int));
        
        num_vertices_all = (int*)malloc(num_graphs * sizeof(int));
        num_edges_all = (int*)malloc(num_graphs * sizeof(int));
        
        for (size_t i = 0; i < num_graphs; ++i) {
            GraphMetaInfo info_local = info[i];
            if (i % local_n == 0) {
                nbr_offs_starts[i/local_n] = nbr_offs_start;
                nbrs_starts[i/local_n] = nbrs_start;
                sizes = 0;
            }
            nbr_offs_local += (info_local.num_vertices + 1);
            nbrs_local += (info_local.num_edges + 1);
            sizes += info_local.num_vertices;
            num_vertices_sum += info_local.num_vertices;
            
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
//        cout << "in process 0, num_vertices_all[11] is: " << num_vertices_all[11] << endl;
    }
    
//    cout << "Process " << my_rank << ": after getting information?" << endl;
    MPI_Bcast(&num_graphs, 1, MPI_INT, 0, comm);
    MPI_Bcast(&num_vertices_sum, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nbr_offs_size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nbrs_size, 1, MPI_INT, 0, comm);
//    cout << "Process " << my_rank << ": " << num_graphs << " " << num_vertices_sum << " " << nbr_offs_size << " " << nbrs_size << " " << num_graphs/num_process << endl; check
    MPI_Barrier(comm);
    local_n = num_graphs/num_process;
//    cout << "my_rank: " << my_rank << " local_n: " << local_n << "\n\n";
    if (my_rank != 0) {
        nbr_offs_starts = (int*)malloc(num_process*sizeof(int));
        nbrs_starts = (int*)malloc(num_process*sizeof(int));
        result_sizes = (int*)malloc(num_process*sizeof(int));
        
        nbr_offs = (int*)malloc(nbr_offs_size*sizeof(int));
        nbrs = (int*)malloc(nbrs_size*sizeof(int));
    }
    
    MPI_Bcast(nbr_offs_starts, num_process, MPI_INT, 0, comm);
    MPI_Bcast(nbrs_starts, num_process, MPI_INT, 0, comm);
    MPI_Bcast(result_sizes, num_process, MPI_INT, 0, comm);
    MPI_Bcast(nbr_offs, nbr_offs_size, MPI_INT, 0, comm);
    MPI_Bcast(nbrs, nbrs_size, MPI_INT, 0, comm);
    
    MPI_Barrier(comm);
    
    local_num_vertices = (int*)malloc(local_n * sizeof(int));
    local_num_edges = (int*)malloc(local_n * sizeof(int));
    
    MPI_Scatter(num_vertices_all, local_n, MPI_INT, local_num_vertices, local_n, MPI_INT, 0, comm);
    MPI_Scatter(num_edges_all, local_n, MPI_INT, local_num_edges, local_n, MPI_INT, 0, comm);
    
    
    int *num_cluster_total_local = nullptr;
    num_cluster_total_local = (int*)malloc(local_n * sizeof(int));
    int *local_results = nullptr;
    local_results = (int*)malloc((result_sizes[my_rank]) * sizeof(int));
    
    nbr_offs_local = nbr_offs + nbr_offs_starts[my_rank];
    nbrs_local = nbrs + nbrs_starts[my_rank];
    
    int index = 0;
    for (size_t i = 0; i < (num_graphs/num_process); ++i) {
        GraphMetaInfo info_local;
        info_local.num_vertices = local_num_vertices[i];
        info_local.num_edges = local_num_edges[i];
        int local_result[info_local.num_vertices];
        int num_cluster_local = clustering(info_local, nbr_offs_local, nbrs_local, local_result);
        for (int j = 0; j < info_local.num_vertices; ++j) {
            local_results[index] = local_result[j];
            index++;
        }
        num_cluster_total_local[i] = num_cluster_local;
        
//        cout << "Process " << my_rank << " graph " << i << " num_cluster_local: " << num_cluster_local << endl;
        
        nbr_offs_local += (info_local.num_vertices + 1);
        nbrs_local += (info_local.num_edges + 1);
    }

    num_cluster_total = (int*)malloc(num_graphs*sizeof(int));
    MPI_Gather(num_cluster_total_local, local_n, MPI_INT, num_cluster_total, local_n, MPI_INT, 0, comm);
    
    MPI_Barrier(comm);

    clustering_results = (int*)malloc(num_vertices_sum*sizeof(int));
    
    int *uneven_all = nullptr;
    int *displs = nullptr;
    int rbuf = 0;
    
    displs = (int*)malloc(num_process*sizeof(int));
    displs[0] = 0;
    rbuf += (result_sizes[0] + 1);
    for (int i = 1; i < num_process; ++i) {
        displs[i] = result_sizes[i-1] + displs[i-1] + 1;
        rbuf += (result_sizes[i] + 1);
    }
    
    uneven_all = (int*)malloc(rbuf*sizeof(int));
    MPI_Gatherv(local_results, result_sizes[my_rank], MPI_INT, uneven_all, result_sizes, displs, MPI_INT, 0, comm);
    
    int j = 0;
    for (int i = 0; i < num_process; ++i) {
        for (int k = displs[i]; k < displs[i] + result_sizes[i]; ++k) {
            clustering_results[j] = uneven_all[k];
            ++j;
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
