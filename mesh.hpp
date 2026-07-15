#ifndef MESH_HPP
#define MESH_HPP

#include <vector>
#include <string>
#include <cstdint>

struct Node
{
    double x, y;
};

struct Triangle
{
    int v1, v2, v3;
    int partition;
};

class Mesh
{
public:
    std::vector<Node> nodes;
    std::vector<Triangle> triangles;

    void partitionMesh(int numParts);

    bool saveBinary(const std::string &filename) const;
    bool loadBinary(const std::string &filename);

    void generateStructuredMesh2D(int targetTriangleCount);

    void shuffleTriangles();
};

#endif // MESH_HPP
