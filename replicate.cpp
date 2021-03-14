//
//  replicate.cpp
//  5112Assignment
//
//  Created by MENG Zihan on 3/14/21.
//
#include "mpi_assignment/clustering.h"

#include <stdio.h>
#include "mpi.h"

#include <cassert>
#include <chrono>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    MPI_Comm comm;
    int num_process;
    int my_rank;
    
    comm = MPI_COMM_WORLD;
    
    MPI_Comm_size(comm, &num_process);
    MPI_Comm_rank(comm, &my_rank);
    
    int local_n;
    //replicate
    int *i_local_results = nullptr;
    int *results = nullptr;
    int uneven_sizes[4] = {1, 2, 3, 4};
    int total_num = 10;
    
    int *uneven_local = nullptr;
    uneven_local = (int*)malloc(uneven_sizes[my_rank]*sizeof(int));
    for (size_t i = 0; i < uneven_sizes[my_rank]; ++i) {
        uneven_local[i] = my_rank;
    }
    
    int *displs = nullptr;
    int rbuf = 0;
    
    displs = (int*)malloc(num_process*sizeof(int));
    displs[0] = 0;
    rbuf += (uneven_sizes[0] + 1);
    for (int i = 1; i < num_process; ++i) {
        displs[i] = uneven_sizes[i-1] + displs[i-1] + 1;
        rbuf += (uneven_sizes[i] + 1);
    }

    int *uneven_all = nullptr;
    uneven_all = (int*)malloc(rbuf*sizeof(int));
    MPI_Gatherv(uneven_local, uneven_sizes[my_rank], MPI_INT, uneven_all, uneven_sizes, displs, MPI_INT, 0, comm);
    
    int *final_result = nullptr;
    final_result = (int*)malloc(total_num*sizeof(int));
    int j = 0;
    for (int i = 0; i < num_process; ++i) {
        for (int k = displs[i]; k < displs[i] + uneven_sizes[i]; ++k) {
            final_result[j] = uneven_all[k];
            ++j;
        }
    }
    
    if (my_rank == 0) {
        for (int i = 0; i < rbuf; ++i) {
            cout << uneven_all[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < rbuf; ++i) {
            cout << final_result[i] << " ";
        }
        cout << endl;
    }
    
    i_local_results = (int*)malloc(local_n*sizeof(int));
    for (size_t i = 0; i < local_n; ++i) {
        i_local_results[i] = my_rank;
    }
    
    for (size_t i = 0; i < local_n; ++i) {
        cout << "process " << my_rank << ": " << i << " " << i_local_results[i] << endl;
    }
    MPI_Barrier(comm);
    results = (int*)malloc(12*sizeof(int));
    MPI_Gather(i_local_results, local_n, MPI_INT, results, local_n, MPI_INT, 0, comm);
    
    if (my_rank == 0) {
        for (int i = 0; i < 12; ++i) {
            cout << results[i] << " ";
        }
        cout << endl;
    }
    //replicate
    
    std::string dir(argv[1]);
    std::string result_path(argv[2]);
    
    int num_graphs;
    int *clustering_results = nullptr;
    int *num_cluster_total = nullptr;
    
    int *nbr_offs = nullptr, *nbrs = nullptr;
    int *nbr_offs_local = nullptr, *nbrs_local = nullptr;
    
    GraphMetaInfo *info = nullptr;
    
    if (my_rank == 0) {
        num_graphs = read_files(dir, info, nbr_offs, nbrs);
    }
    
    int *nbr_offs_starts = nullptr;
    int *nbrs_starts = nullptr;
    int *result_sizes = nullptr;
    int *num_vertices_all = nullptr;
    int *num_edges_all = nullptr;
    
    
    
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
    
    local_n = 12/num_process;
    
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
    }
    
    MPI_Bcast(&num_graphs, 1, MPI_INT, 0, comm);
    MPI_Bcast(&num_vertices_sum, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nbr_offs_size, 1, MPI_INT, 0, comm);
    MPI_Bcast(&nbrs_size, 1, MPI_INT, 0, comm);
    local_n = num_graphs/num_process;
    
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
        
        cout << "Process " << my_rank << " graph " << i << " num_cluster_local: " << num_cluster_local << endl;
        
        nbr_offs_local += (info_local.num_vertices + 1);
        nbrs_local += (info_local.num_edges + 1);
    }
    
    num_cluster_total = (int*)malloc(num_graphs*sizeof(int));
    MPI_Gather(num_cluster_total_local, local_n, MPI_INT, num_cluster_total, local_n, MPI_INT, 0, comm);
    if (my_rank == 0) {
        cout << "num_cluster_total[size-1]: " << num_cluster_total[11] << endl;;
    }
    
//    clustering_results = (int*)malloc(num_vertices_sum*sizeof(int));
//    cout << "result_sizes[my_rank]" << result_sizes[my_rank] << endl;
//    cout << "num_vertices_sum " << num_vertices_sum << endl;
//    MPI_Gather(local_results, result_sizes[my_rank], MPI_INT, clustering_results, result_sizes[my_rank], MPI_INT, 0, comm);
//
//    if (my_rank == 0) {
//        cout << "clustering_results[0]: " << clustering_results[0] << endl;
//    }

    // pseudo code
    MPI_Barrier(comm);
    MPI_Finalize();
}
