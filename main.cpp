#include "mesh.hpp"
#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

// pair (triangle, morton)
struct TriangleWithMorton
{
    Triangle tri;
    uint64_t morton;
};

// Morton calculatioon helper function
static inline uint64_t interleaveBits(uint32_t x, uint32_t y)
{
    uint64_t xx = x, yy = y;
    xx = (xx | (xx << 16)) & 0x0000FFFF0000FFFFULL;
    xx = (xx | (xx << 8)) & 0x00FF00FF00FF00FFULL;
    xx = (xx | (xx << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    xx = (xx | (xx << 2)) & 0x3333333333333333ULL;
    xx = (xx | (xx << 1)) & 0x5555555555555555ULL;

    yy = (yy | (yy << 16)) & 0x0000FFFF0000FFFFULL;
    yy = (yy | (yy << 8)) & 0x00FF00FF00FF00FFULL;
    yy = (yy | (yy << 4)) & 0x0F0F0F0F0F0F0F0FULL;
    yy = (yy | (yy << 2)) & 0x3333333333333333ULL;
    yy = (yy | (yy << 1)) & 0x5555555555555555ULL;

    return (xx << 1) | yy;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mesh fullMesh;
    std::vector<Node> allNodes;
    std::vector<Triangle> localTriangles;

    MPI_Win nodeWin;
    size_t nodeCount = 0;
    uint64_t globalTriCount = 0;

    // Mesh building and loading
    if (rank == 0)
    {
        std::cerr << "[Rank 0] Starting with " << size << " processors\n";
        // parse mesh size
        uint64_t meshTarget = 50000;
        if (argc >= 2)
        {
            try
            {
                meshTarget = static_cast<uint64_t>(std::stoull(argv[1]));
            }
            catch (...)
            {
                std::cerr << "[Rank 0] Invalid mesh_size; using default 50000\n";
            }
        }
        else
        {
            std::cout << "[Rank 0] Usage: mpirun -np P ./partitioner <mesh_size>\n"
                         "[Rank 0] Defaulting mesh_size=50000\n";
        }

        const std::string meshFile = "mesh_" + std::to_string(meshTarget);
        bool loaded = false;
        if (std::filesystem::exists(meshFile))
        {
            loaded = fullMesh.loadBinary(meshFile);
            if (!loaded)
                std::cerr << "[Rank 0] Found '" << meshFile << "' but failed to load; regenerating\n";
        }
        if (!loaded)
        {
            std::cout << "[Rank 0] Generating ~" << meshTarget << " triangles...\n";
            fullMesh.generateStructuredMesh2D(static_cast<int>(meshTarget));
            fullMesh.shuffleTriangles();
            if (!fullMesh.saveBinary(meshFile))
                std::cerr << "[Rank 0] Warning: failed to save '" << meshFile << "'\n";
            else
                std::cout << "[Rank 0] Saved '" << meshFile << "'\n";
        }
        else
        {
            std::cout << "[Rank 0] Loaded '" << meshFile << "'\n";
        }

        allNodes = fullMesh.nodes;
        nodeCount = allNodes.size();
        globalTriCount = static_cast<uint64_t>(fullMesh.triangles.size());
    }

    // broadcast basic counts so all ranks can size buffers
    MPI_Bcast(&nodeCount, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalTriCount, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    // Triangle distribution
    std::vector<int> triCounts(size, 0), triDispls(size, 0);
    if (rank == 0)
    {
        const size_t total = fullMesh.triangles.size();
        const size_t perRank = (total + size - 1) / size;
        for (int r = 0; r < size; ++r)
        {
            const size_t start = static_cast<size_t>(r) * perRank;
            const size_t end = std::min(start + perRank, total);
            const size_t cnt = (end > start ? end - start : 0);
            triCounts[r] = static_cast<int>(cnt);
        }
        int d = 0;
        for (int r = 0; r < size; ++r)
        {
            triDispls[r] = d;
            d += triCounts[r];
        }
    }

    // Let each rank learn its recv count
    int myCount = 0;
    MPI_Scatter(triCounts.data(), 1, MPI_INT, &myCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    localTriangles.resize(static_cast<size_t>(myCount));

    // Create an MPI datatype for Triangle (packed blob)
    MPI_Datatype TRI;
    MPI_Type_contiguous(static_cast<int>(sizeof(Triangle)), MPI_BYTE, &TRI);
    MPI_Type_commit(&TRI);

    // Scatter the triangle array
    if (rank == 0)
    {
        MPI_Scatterv(fullMesh.triangles.data(), triCounts.data(), triDispls.data(), TRI,
                     localTriangles.data(), myCount, TRI, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(nullptr, nullptr, nullptr, TRI,
                     localTriangles.data(), myCount, TRI, 0, MPI_COMM_WORLD);
    }

    // Weuse an RMA window here because it makes no sense for each processor to hold a copy.
    if (rank == 0)
    {
        MPI_Win_create(allNodes.data(),
                       static_cast<MPI_Aint>(nodeCount * sizeof(Node)),
                       sizeof(Node), MPI_INFO_NULL, MPI_COMM_WORLD, &nodeWin);
    }
    else
    {
        allNodes.resize(nodeCount);
        MPI_Win_create(nullptr, 0, sizeof(Node), MPI_INFO_NULL, MPI_COMM_WORLD, &nodeWin);
    }

    if (rank != 0)
    {
        MPI_Win_fence(0, nodeWin);
        MPI_Get(allNodes.data(), static_cast<int>(nodeCount * sizeof(Node)), MPI_BYTE,
                0, 0, static_cast<int>(nodeCount * sizeof(Node)), MPI_BYTE, nodeWin);
        MPI_Win_fence(0, nodeWin);
    }
    else
    {
        MPI_Win_fence(0, nodeWin);
        MPI_Win_fence(0, nodeWin);
    }

    // local sort + morton calculation
    std::vector<TriangleWithMorton> localWithCodes;
    localWithCodes.reserve(localTriangles.size());
    for (const auto &tri : localTriangles)
    {
        const Node &A = allNodes[tri.v1];
        const Node &B = allNodes[tri.v2];
        const Node &C = allNodes[tri.v3];
        double cx = (A.x + B.x + C.x) / 3.0;
        double cy = (A.y + B.y + C.y) / 3.0;
        uint32_t ix = static_cast<uint32_t>(cx * ((1u << 16) - 1));
        uint32_t iy = static_cast<uint32_t>(cy * ((1u << 16) - 1));
        localWithCodes.push_back(TriangleWithMorton{tri, interleaveBits(ix, iy)});
    }
    std::sort(localWithCodes.begin(), localWithCodes.end(),
              [](const auto &a, const auto &b)
              { return a.morton < b.morton; });

    // Pivots for sample sort
    std::vector<uint64_t> localSamples(std::max(0, size - 1), 0);
    if (!localWithCodes.empty() && size > 1)
    {
        const size_t stride = std::max<size_t>(1, localWithCodes.size() / size);
        for (int i = 1; i < size; ++i)
        {
            const size_t idx = std::min(localWithCodes.size() - 1, static_cast<size_t>(i) * stride);
            localSamples[static_cast<size_t>(i - 1)] = localWithCodes[idx].morton;
        }
    }

    std::vector<uint64_t> allSamples(std::max(0, size - 1) * size, 0);
    MPI_Allgather(localSamples.data(), std::max(0, size - 1), MPI_UINT64_T,
                  allSamples.data(), std::max(0, size - 1), MPI_UINT64_T,
                  MPI_COMM_WORLD);

    std::vector<uint64_t> splitters(std::max(0, size - 1), 0);
    if (rank == 0 && size > 1)
    {
        std::sort(allSamples.begin(), allSamples.end());
        for (int i = 1; i < size; ++i)
        {
            size_t q = static_cast<size_t>(i) * (allSamples.size() / size);
            if (q >= allSamples.size())
                q = allSamples.size() - 1;
            splitters[static_cast<size_t>(i - 1)] = allSamples[q];
        }
    }
    if (size > 1)
    {
        MPI_Bcast(splitters.data(), static_cast<int>(splitters.size()), MPI_UINT64_T, 0, MPI_COMM_WORLD);
    }

    // bucketing and all to all
    // blobbing
    MPI_Datatype TRIWM;
    MPI_Type_contiguous(static_cast<int>(sizeof(TriangleWithMorton)), MPI_BYTE, &TRIWM);
    MPI_Type_commit(&TRIWM);

    std::vector<std::vector<TriangleWithMorton>> buckets(size);
    if (size > 1)
    {
        for (auto &t : localWithCodes)
        {
            int target = static_cast<int>(
                std::lower_bound(splitters.begin(), splitters.end(), t.morton) - splitters.begin());
            buckets[static_cast<size_t>(target)].push_back(t);
        }
    }
    else
    {
        buckets[0] = std::move(localWithCodes);
    }

    // counts
    std::vector<int> sendCounts(size, 0), recvCounts(size, 0);
    for (int i = 0; i < size; ++i)
        sendCounts[i] = static_cast<int>(buckets[static_cast<size_t>(i)].size());

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sdispls(size, 0), rdispls(size, 0);
    int sendTotal = 0, recvTotal = 0;
    for (int i = 0; i < size; ++i)
    {
        sdispls[i] = sendTotal;
        sendTotal += sendCounts[i];
        rdispls[i] = recvTotal;
        recvTotal += recvCounts[i];
    }

    std::vector<TriangleWithMorton> sendFlat;
    sendFlat.reserve(static_cast<size_t>(sendTotal));
    for (int i = 0; i < size; ++i)
    {
        auto &bin = buckets[static_cast<size_t>(i)];
        if (!bin.empty())
            sendFlat.insert(sendFlat.end(), bin.begin(), bin.end());
    }

    // sanity check
    int localElems = static_cast<int>(sendFlat.size());
    int sumSend = 0, sumRecv = 0;
    for (int c : sendCounts)
        sumSend += c;
    for (int c : recvCounts)
        sumRecv += c;
    if (sumSend != localElems)
    {
        std::cerr << "[Rank " << rank << "] ERROR: sendCounts sum (" << sumSend
                  << ") != flattened size (" << localElems << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<TriangleWithMorton> owned(static_cast<size_t>(recvTotal));
    MPI_Alltoallv(sendFlat.data(), sendCounts.data(), sdispls.data(), TRIWM,
                  owned.data(), recvCounts.data(), rdispls.data(), TRIWM,
                  MPI_COMM_WORLD);

    std::sort(owned.begin(), owned.end(),
              [](const auto &a, const auto &b)
              { return a.morton < b.morton; });

    // Global conservation check
    int before = localElems, after = static_cast<int>(owned.size());
    int globalBefore = 0, globalAfter = 0;
    MPI_Allreduce(&before, &globalBefore, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&after, &globalAfter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (globalBefore != globalAfter)
    {
        if (rank == 0)
            std::cerr << "ERROR: redistributed total " << globalAfter << " != original total " << globalBefore << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    double t1 = MPI_Wtime();
    double local = t1 - t0;

    if (rank == 0)
    {
        std::cout << "\nRank 0 reports time of " << local << "\n";
    }

    // Visualization code for small meshes. Output file to be read by visual.py
    if (globalTriCount < 1000001)
    {
        int localOwned = static_cast<int>(owned.size());
        std::vector<int> counts(size, 0);
        MPI_Gather(&localOwned, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs, recvCountsEl;
        int totalRecvEl = 0;
        if (rank == 0)
        {
            recvCountsEl = counts;
            displs.resize(size, 0);
            for (int r = 0; r < size; ++r)
            {
                displs[r] = totalRecvEl;
                totalRecvEl += recvCountsEl[r];
            }
        }

        std::vector<TriangleWithMorton> gathered(static_cast<size_t>(rank == 0 ? totalRecvEl : 0));
        MPI_Gatherv(owned.data(), localOwned, TRIWM,
                    gathered.data(), recvCountsEl.data(), displs.data(), TRIWM,
                    0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::ofstream out("partitions.txt");
            for (int r = 0; r < size; ++r)
            {
                const int start = displs[r];
                const int cnt = recvCountsEl[r];
                for (int i = 0; i < cnt; ++i)
                {
                    const Triangle &t = gathered[static_cast<size_t>(start + i)].tri;
                    const Node &A = allNodes[t.v1];
                    const Node &B = allNodes[t.v2];
                    const Node &C = allNodes[t.v3];
                    out << A.x << " " << A.y << "  "
                        << B.x << " " << B.y << "  "
                        << C.x << " " << C.y << "  "
                        << r << "\n"; // partition = owning rank
                }
            }
            out.close();
            std::cout << "[Rank 0] Wrote partitions.txt (" << globalTriCount << " triangles)\n";
        }
    }

    MPI_Type_free(&TRIWM);
    MPI_Type_free(&TRI);
    MPI_Win_free(&nodeWin);
    MPI_Finalize();
    return 0;
}
