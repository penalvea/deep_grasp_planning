#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sstream>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>


int main(int argc, char** argv){
  int size_mat=std::atoi(argv[1]);


  std::ifstream file;
  std::cout<<argv[2]<<std::endl;
  file.open(argv[2]);
  if(!file){
    std::cout<<"error al abrir el archivo"<<std::endl;
    return -1;
  }

  file.seekg(0, file.end);
  int length=file.tellg();
  file.seekg(0, file.beg);

  char *buffer=new char[length];

  file.read(buffer, length);
  if (file)
    std::cout << "all characters read successfully."<<std::endl;
  else{
    std::cout << "error: only " << file.gcount() << " could be read"<<std::endl;
    return -1;
  }





  std::vector< std::vector <std::vector< int > > > mat;
  for(int l=0; l<size_mat; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<size_mat; m++){
      std::vector<int> vec;
      for(int n=0; n<size_mat; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    mat.push_back(vec_vec);
  }


  std::stringstream ss;
  ss.str(buffer);
  std::string item;


  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){

        std::getline(ss, item, ' ' );
        if(item=="1")
          mat[i][j][k]=1;
      }
    }
  }









  std::ifstream file2;
  std::cout<<argv[3]<<std::endl;
  file2.open(argv[3]);
  if(!file2){
    std::cout<<"error al abrir el archivo"<<std::endl;
    return -1;
  }

  file2.seekg(0, file2.end);
  int length2=file2.tellg();
  file2.seekg(0, file2.beg);

  char *buffer2=new char[length2];

  file2.read(buffer2, length2);
  if (file2)
    std::cout << "all characters read successfully."<<std::endl;
  else{
    std::cout << "error: only " << file2.gcount() << " could be read"<<std::endl;
    return -1;
  }



  std::vector< std::vector <std::vector< int > > > mat2;
  for(int l=0; l<size_mat; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<size_mat; m++){
      std::vector<int> vec;
      for(int n=0; n<size_mat; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    mat2.push_back(vec_vec);
  }



  std::stringstream ss2;
  ss2.str(buffer2);
  std::string item2;


  for(int i=0; i<mat2.size(); i++){
    for(int j=0; j<mat2[0].size(); j++){
      for(int k=0; k<mat2[0][0].size(); k++){

        std::getline(ss2, item2, ' ' );
        if(item2=="1")
          mat2[i][j][k]=1;
      }
    }
  }






  std::ifstream file3;
  std::cout<<argv[4]<<std::endl;
  file3.open(argv[4]);
  if(!file3){
    std::cout<<"error al abrir el archivo"<<std::endl;
    return -1;
  }

  file3.seekg(0, file3.end);
  int length3=file3.tellg();
  file3.seekg(0, file3.beg);

  char *buffer3=new char[length3];

  file3.read(buffer3, length3);
  if (file3)
    std::cout << "all characters read successfully."<<std::endl;
  else{
    std::cout << "error: only " << file3.gcount() << " could be read"<<std::endl;
    return -1;
  }



  std::vector< std::vector <std::vector< int > > > mat3;
  for(int l=0; l<size_mat; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<size_mat; m++){
      std::vector<int> vec;
      for(int n=0; n<size_mat; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    mat3.push_back(vec_vec);
  }



  std::stringstream ss3;
  ss3.str(buffer3);
  std::string item3;


  for(int i=0; i<mat3.size(); i++){
    for(int j=0; j<mat3[0].size(); j++){
      for(int k=0; k<mat3[0][0].size(); k++){

        std::getline(ss3, item3, ' ' );
        if(item3=="1")
          mat3[i][j][k]=1;
      }
    }
  }






  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0,0,0);
  viewer->addCoordinateSystem(50.0);
  viewer->initCameraParameters();

  int cont=0;
  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          std::ostringstream name;
          name<<cont;
          viewer->addCube(i, i+1, j, j+1, k, k+1, 1.0, 0.0, 0.0, name.str());
          cont++;
        }
        if(mat2[i][j][k]==1 && mat[i][j][k]==0){
          std::ostringstream name;
          name<<cont;
          viewer->addCube(i, i+1, j, j+1, k, k+1, 0.0, 1.0, 0.0, name.str());
          cont++;
        }
        if(mat3[i][j][k]==1 && mat2[i][j][k]==0 && mat[i][j][k]==0){
          std::ostringstream name;
          name<<cont;
          viewer->addCube(i, i+1, j, j+1, k, k+1, 0.0, 0.0, 1.0, name.str());
          cont++;
        }
      }
    }
  }

  while(!viewer->wasStopped()){
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }

  return 0;
}
