#include "vtkAutoInit.h" 
VTK_MODULE_INIT(vtkRenderingOpenGL); // VTK was built with vtkRenderingOpenGL2
VTK_MODULE_INIT(vtkInteractionStyle);

#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "Config.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <vtkVersion.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkTriangle.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

string removeSuffix(string fileName) {
    const char* full_name = fileName.c_str();
    const char* mn_first = full_name;
    const char* mn_last = full_name;

    if (strrchr(full_name, '.') != NULL)
        mn_last = strrchr(full_name, '.');  //获取.*后缀

    fileName.assign(mn_first, mn_last);
    return fileName;
}

string getSuffix(string fileName) {
    const char* full_name = fileName.c_str();
    const char* mn_first = full_name;
    const char* mn_last = full_name;

    if (strrchr(full_name, '.') != NULL)
        mn_last = strrchr(full_name, '.') + 1;  //获取.*后缀
    fileName.assign(mn_last);
    return fileName;
}

inline double
uniform_deviate(int seed)
{
    double ran = seed * (1.0 / (RAND_MAX + 1.0));
    return ran;
}

inline void
randomPointTriangle(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3,
    Eigen::Vector4f& p)
{
    float r1 = static_cast<float> (uniform_deviate(rand()));
    float r2 = static_cast<float> (uniform_deviate(rand()));
    float r1sqr = std::sqrt(r1);
    float OneMinR1Sqr = (1 - r1sqr);
    float OneMinR2 = (1 - r2);
    a1 *= OneMinR1Sqr;
    a2 *= OneMinR1Sqr;
    a3 *= OneMinR1Sqr;
    b1 *= OneMinR2;
    b2 *= OneMinR2;
    b3 *= OneMinR2;
    c1 = r1sqr * (r2 * c1 + b1) + a1;
    c2 = r1sqr * (r2 * c2 + b2) + a2;
    c3 = r1sqr * (r2 * c3 + b3) + a3;
    p[0] = c1;
    p[1] = c2;
    p[2] = c3;
    p[3] = 0;
}

inline void
randPSurface(vtkPolyData* polydata, std::vector<double>* cumulativeAreas, double totalArea, Eigen::Vector4f& p, bool calcNormal, Eigen::Vector3f& n)
{
    float r = static_cast<float> (uniform_deviate(rand()) * totalArea);

    std::vector<double>::iterator low = std::lower_bound(cumulativeAreas->begin(), cumulativeAreas->end(), r);
    vtkIdType el = vtkIdType(low - cumulativeAreas->begin());

    double A[3], B[3], C[3];
    vtkIdType npts = 0;
    vtkIdType* ptIds = NULL;
    polydata->GetCellPoints(el, npts, ptIds);
    polydata->GetPoint(ptIds[0], A);
    polydata->GetPoint(ptIds[1], B);
    polydata->GetPoint(ptIds[2], C);
    if (calcNormal)
    {
        // OBJ: Vertices are stored in a counter-clockwise order by default
        Eigen::Vector3f v1 = Eigen::Vector3f(A[0], A[1], A[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
        Eigen::Vector3f v2 = Eigen::Vector3f(B[0], B[1], B[2]) - Eigen::Vector3f(C[0], C[1], C[2]);
        n = v1.cross(v2);
        n.normalize();
    }
    randomPointTriangle(float(A[0]), float(A[1]), float(A[2]),
        float(B[0]), float(B[1]), float(B[2]),
        float(C[0]), float(C[1]), float(C[2]), p);
}

void
uniform_sampling(vtkSmartPointer<vtkPolyData> polydata, size_t n_samples, bool calc_normal, pcl::PointCloud<pcl::PointNormal>& cloud_out)
{
    polydata->BuildCells();
    vtkSmartPointer<vtkCellArray> cells = polydata->GetPolys();

    double p1[3], p2[3], p3[3], totalArea = 0;
    std::vector<double> cumulativeAreas(cells->GetNumberOfCells(), 0);
    size_t i = 0;
    vtkIdType npts = 0, *ptIds = NULL;
    for (cells->InitTraversal(); cells->GetNextCell(npts, ptIds); i++)
    {
        polydata->GetPoint(ptIds[0], p1);
        polydata->GetPoint(ptIds[1], p2);
        polydata->GetPoint(ptIds[2], p3);
        totalArea += vtkTriangle::TriangleArea(p1, p2, p3);
        cumulativeAreas[i] = totalArea;
    }

    cloud_out.points.resize(n_samples);
    cloud_out.width = static_cast<pcl::uint32_t> (n_samples);
    cloud_out.height = 1;

    for (i = 0; i < n_samples; i++)
    {
        Eigen::Vector4f p;
        Eigen::Vector3f n;
        randPSurface(polydata, &cumulativeAreas, totalArea, p, calc_normal, n);
        cloud_out.points[i].x = p[0];
        cloud_out.points[i].y = p[1];
        cloud_out.points[i].z = p[2];
        if (calc_normal)
        {
            cloud_out.points[i].normal_x = n[0];
            cloud_out.points[i].normal_y = n[1];
            cloud_out.points[i].normal_z = n[2];
        }
    }
}

int main()
{
    // Parse command line arguments
    map<string, string> serial_cfg;
    ReadConfig("config.txt", serial_cfg);
    string input = serial_cfg["input"];
    int sp = atof(serial_cfg["number_samples"].c_str());
    float leaf_size = atof(serial_cfg["leaf_size"].c_str());
    bool vis_result = atof(serial_cfg["vis_result"].c_str());
    const bool write_normals = atof(serial_cfg["write_normals"].c_str());
    string suffix = getSuffix(input);
    string output = removeSuffix(input) + ".pcd";

    cout << "input = " << input << endl;
    cout << "output = " << output << endl;
    cout << "sample_number = " << sp << endl;
    cout << "vis_result = " << vis_result << endl;
    cout << "write_normals = " << write_normals << endl;

    vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
    if (suffix == "ply")
    {
        pcl::PolygonMesh mesh;
        pcl::io::loadPolygonFilePLY(input, mesh);
        pcl::io::mesh2vtk(mesh, polydata1);
    }
    else if (suffix == "obj")
    {
        vtkSmartPointer<vtkOBJReader> readerQuery = vtkSmartPointer<vtkOBJReader>::New();
        readerQuery->SetFileName(input.c_str());
        readerQuery->Update();
        polydata1 = readerQuery->GetOutput();
    }

    //make sure that the polygons are triangles!
    vtkSmartPointer<vtkTriangleFilter> triangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
#if VTK_MAJOR_VERSION < 6
    triangleFilter->SetInput(polydata1);
#else
    triangleFilter->SetInputData(polydata1);
#endif
    triangleFilter->Update();

    vtkSmartPointer<vtkPolyDataMapper> triangleMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    triangleMapper->SetInputConnection(triangleFilter->GetOutputPort());
    triangleMapper->Update();
    polydata1 = triangleMapper->GetInput();

    bool INTER_VIS = false;

    if (INTER_VIS)
    {
        visualization::PCLVisualizer vis;
        vis.addModelFromPolyData(polydata1, "mesh1", 0);
        vis.setRepresentationToSurfaceForAllActors();
        vis.spin();
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_1(new pcl::PointCloud<pcl::PointNormal>);
    uniform_sampling(polydata1, sp, write_normals, *cloud_1);

    if (INTER_VIS)
    {
        visualization::PCLVisualizer vis_sampled;
        vis_sampled.addPointCloud<pcl::PointNormal>(cloud_1);
        if (write_normals)
            vis_sampled.addPointCloudNormals<pcl::PointNormal>(cloud_1, 1, 0.02f, "cloud_normals");
        vis_sampled.spin();
    }

    // Voxelgrid
    VoxelGrid<PointNormal> grid_;
    grid_.setInputCloud(cloud_1);
    grid_.setLeafSize(leaf_size, leaf_size, leaf_size);

    pcl::PointCloud<pcl::PointNormal>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointNormal>);
    grid_.filter(*voxel_cloud);

    if (vis_result)
    {
        visualization::PCLVisualizer vis3("VOXELIZED SAMPLES CLOUD");
        vis3.addPointCloud<pcl::PointNormal>(voxel_cloud);
        if (write_normals)
            vis3.addPointCloudNormals<pcl::PointNormal>(voxel_cloud, 1, 0.02f, "cloud_normals");
        vis3.spin();
    }

    if (!write_normals)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        // Strip uninitialized normals from cloud:
        pcl::copyPointCloud(*voxel_cloud, *cloud_xyz);
        savePCDFileASCII(output, *cloud_xyz);
    }
    else
    {
        savePCDFileASCII(output, *voxel_cloud);
    }
}