# Parallel Mesh Partitioning

A C++ and MPI implementation of geometric mesh partitioning using Morton codes, also known as Z-order curves.

The program assigns mesh elements to balanced partitions while preserving spatial locality. It was deployed and benchmarked on the University at Buffalo HPC environment using Slurm-managed resources provisioned through ColdFront.

## How It Works

1. Compute the center point of each triangular mesh element.
2. Convert each center point into a Morton code by interleaving its coordinate bits.
3. Sort the mesh elements by Morton code.
4. Divide the sorted elements into approximately equal-sized partitions.
5. Distribute the resulting partitions across MPI processes.

Elements that are close together in physical space usually receive similar Morton codes, helping each partition remain spatially compact.

## Parallel Implementation

The parallel version distributes mesh data across MPI processes and uses:

- MPI for distributed-memory communication
- Parallel sample sort for ordering Morton codes
- MPI one-sided communication for accessing distributed vertex-coordinate data
- Slurm for launching jobs on the HPC cluster

Two approaches were explored:

1. Distributing mesh elements while keeping the vertex-coordinate list available to each process
2. Distributing both mesh elements and vertex-coordinate data across processes