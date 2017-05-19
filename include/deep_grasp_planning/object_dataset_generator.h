#ifndef OBJECT_DATASET_GENERATOR_H
#define OBJECT_DATASET_GENERATOR_H


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>

class ObjectDatasetGenerator
{
  int cont_;
  std::ofstream general_;
  std::string folder_;
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  int side_matrix_;
  int x_max_, x_min_, y_max_, y_min_, z_max_, z_min_, height_max_, height_min_, radius_max_, radius_min_;
public:
  ObjectDatasetGenerator(std::string general_file, std::__cxx11::string folder, int side_matrix);
  void generateDataset(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator);
  void generateDatasetNoCamera(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator);
  void generateDatasetHollowObject(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator);
  void change_sizes(int x_max, int x_min, int y_max, int y_min, int z_max, int z_min, int height_max, int height_min, int radius_max, int radius_min);

private:
  pcl::PointCloud<pcl::PointXYZ>::Ptr generateCube(float x, float y, float z);
  pcl::PointCloud<pcl::PointXYZ>::Ptr generateCylinder(float radius, float height);
  pcl::PointCloud<pcl::PointXYZ>::Ptr generateCone(float radius, float height);
  pcl::PointCloud<pcl::PointXYZ>::Ptr generateSphere(float radius);
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rot_x, float rot_y, float rot_z);
  pcl::PointCloud<pcl::PointXYZ>::Ptr translatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float trans_x, float trans_y, float trans_z);
  std::vector< std::vector <std::vector< int > > > getMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  std::vector< std::vector <std::vector< int > > > getSideMatrix(std::vector< std::vector <std::vector< int > > > mat);
  std::vector< std::vector <std::vector< int > > > getHollowMatrix(std::vector< std::vector <std::vector< int > > > mat);
  std::vector<int> getDisplacement(std::vector< std::vector <std::vector< int > > > mat);
  std::vector< std::vector <std::vector< int > > > moveMatrix(std::vector< std::vector <std::vector< int > > > mat, std::vector<int> disp);
  int visualizeMat(std::vector< std::vector <std::vector< int > > > mat, float r, float g, float b, int start);
  void visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string name);
  pcl::PointCloud<pcl::PointXYZ>::Ptr getSidePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated, std::vector<std::vector<std::vector<int> > > side_mat);
  std::vector< std::vector< std::vector <std::vector< int > > > > generateMats(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z, float camera_rot_x, float camera_rot_y, float camera_rot_z, float camera_trans_x, float camera_trans_y, float camera_trans_z);
  std::vector< std::vector< std::vector <std::vector< int > > > > generateMatsNoCamera(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z);
  std::vector< std::vector< std::vector <std::vector< int > > > > generateMatsNoCameraHollowObject(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z);
  void writeMat(std::vector< std::vector <std::vector< int > > > mat, const std::string file_name);




};

#endif // OBJECT_DATASET_GENERATOR_H
