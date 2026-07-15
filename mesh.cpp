#include "mesh.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <random>

// Helper to interleave bits , we dont use this in the parallel version anymore. Its in the main now
static inline uint64_t interleaveBits(uint32_t x, uint32_t y)
{

    uint64_t xx = x;
    uint64_t yy = y;
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

// SEQUENTIAL
void Mesh::partitionMesh(int numParts)
{
    if (triangles.empty() || nodes.empty())
        return;

    double minX = nodes[0].x, maxX = nodes[0].x;
    double minY = nodes[0].y, maxY = nodes[0].y;
    for (const auto &n : nodes)
    {
        if (n.x < minX)
            minX = n.x;
        if (n.x > maxX)
            maxX = n.x;
        if (n.y < minY)
            minY = n.y;
        if (n.y > maxY)
            maxY = n.y;
    }
    double scaleX = (maxX - minX) > 0 ? (1.0 / (maxX - minX)) : 1.0;
    double scaleY = (maxY - minY) > 0 ? (1.0 / (maxY - minY)) : 1.0;
    struct TriWithCode
    {
        Triangle *tri;
        uint64_t morton;
    };
    std::vector<TriWithCode> coded;
    coded.reserve(triangles.size());

    for (auto &tri : triangles)
    {
        const Node &A = nodes[tri.v1];
        const Node &B = nodes[tri.v2];
        const Node &C = nodes[tri.v3];
        double cx = (A.x + B.x + C.x) / 3.0;
        double cy = (A.y + B.y + C.y) / 3.0;

        // normalize
        double nx = (cx - minX) * scaleX;
        double ny = (cy - minY) * scaleY;

        uint32_t ix = static_cast<uint32_t>(nx * ((1u << 16) - 1));
        uint32_t iy = static_cast<uint32_t>(ny * ((1u << 16) - 1));

        uint64_t mortonCode = interleaveBits(ix, iy);
        coded.push_back({&tri, mortonCode});
    }

    std::sort(coded.begin(), coded.end(),
              [](const TriWithCode &a, const TriWithCode &b)
              {
                  return a.morton < b.morton;
              });

    size_t N = coded.size();
    size_t perPart = (N + numParts - 1) / numParts;
    for (size_t i = 0; i < N; ++i)
    {
        int part = static_cast<int>(i / perPart);
        if (part >= numParts)
            part = numParts - 1;
        coded[i].tri->partition = part;
    }
}

bool Mesh::saveBinary(const std::string &filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        return false;
    uint64_t n = nodes.size(), t = triangles.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(nodes.data()), n * sizeof(Node));
    out.write(reinterpret_cast<const char *>(&t), sizeof(t));
    out.write(reinterpret_cast<const char *>(triangles.data()), t * sizeof(Triangle));
    return true;
}

bool Mesh::loadBinary(const std::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        return false;
    uint64_t n = 0, t = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    nodes.resize(n);
    in.read(reinterpret_cast<char *>(nodes.data()), n * sizeof(Node));
    in.read(reinterpret_cast<char *>(&t), sizeof(t));
    triangles.resize(t);
    in.read(reinterpret_cast<char *>(triangles.data()), t * sizeof(Triangle));
    return true;
}

void Mesh::generateStructuredMesh2D(int targetTriangleCount)
{
    triangles.clear();
    nodes.clear();

    int approxSquares = targetTriangleCount / 2;
    int Nx = static_cast<int>(std::sqrt(approxSquares));
    int Ny = (Nx == 0) ? 1 : (approxSquares / Nx);
    if (Nx == 0 || Ny == 0)
        Nx = Ny = 1;

    double dx = 1.0 / Nx;
    double dy = 1.0 / Ny;

    for (int j = 0; j <= Ny; ++j)
    {
        for (int i = 0; i <= Nx; ++i)
        {
            nodes.push_back(Node{i * dx, j * dy});
        }
    }

    auto idx = [Nx](int i, int j)
    { return j * (Nx + 1) + i; };

    for (int j = 0; j < Ny; ++j)
    {
        for (int i = 0; i < Nx; ++i)
        {
            int v0 = idx(i, j);
            int v1 = idx(i + 1, j);
            int v2 = idx(i, j + 1);
            int v3 = idx(i + 1, j + 1);

            triangles.push_back(Triangle{v0, v1, v3, 0});
            triangles.push_back(Triangle{v0, v3, v2, 0});
        }
    }
}

void Mesh::shuffleTriangles()
{
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::shuffle(triangles.begin(), triangles.end(), rng);
}
