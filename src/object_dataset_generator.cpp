#include "deep_grasp_planning/object_dataset_generator.h"

ObjectDatasetGenerator::ObjectDatasetGenerator(std::string general_file, std::string folder, int side_matrix)
{
  cont_=0;
  std:srand(std::time(0));
  general_.open(general_file.c_str(), std::ofstream::out | std::ofstream::trunc);
  general_<<"id shape training/validation x/radius y/height z rot_x rot_y rot_z camera_trans_x camera_trans_y camera_trans_z camera_rot_x camera_rot_y camera_rot_z\n";
  folder_=folder;
  side_matrix_=side_matrix;

  x_max_=20-4;
  x_min_= 4;
  y_max_=20-4;
  y_min_=4;
  z_max_=15-4;
  z_min_=4;
  height_max_=20-4;
  height_min_=4;
  radius_max_=8-2;
  radius_min_=2;
}

void ObjectDatasetGenerator::change_sizes(int x_max, int x_min, int y_max, int y_min, int z_max, int z_min, int height_max, int height_min, int radius_max, int radius_min){
  x_max_=x_max-x_min;
  x_min_= x_min;
  y_max_=y_max-y_min;
  y_min_=y_min;
  z_max_=z_max-z_min;
  z_min_=z_min;
  height_max_=height_max-height_min;
  height_min_=height_min;
  radius_max_=radius_max-radius_min;
  radius_min_=radius_min;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::generateCube(float x, float y, float z){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->height=1;
  int cont=0;
  for(float i=-x/2; i<x/2; i+=0.25){
    for(float j=-y/2; j<y/2; j+=0.25){
      for(float k=-z/2; k<z/2; k+=0.25){
        cont++;
        cloud->width=cont;
        cloud->points.resize(cont);
        cloud->points[cont-1].x=side_matrix_/2+i;
        cloud->points[cont-1].y=side_matrix_/2+j;
        cloud->points[cont-1].z=side_matrix_/2+k;
      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::generateCylinder(float radius, float height){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width=1;
  int cont=0;
  for(float h=-height/2; h<height/2; h+=0.25){
    for(float i=0; i<side_matrix_; i+=0.25){
      for(float j=0; j<side_matrix_; j+=0.25){
        if(std::sqrt(((i-side_matrix_/2)*(i-side_matrix_/2))+((j-side_matrix_/2)*(j-side_matrix_/2)))<=radius){
          cont++;
          cloud->height=cont;
          cloud->points.resize(cont);
          cloud->points[cont-1].x=i;
          cloud->points[cont-1].y=j;
          cloud->points[cont-1].z=side_matrix_/2+h;
        }

      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::generateCone(float radius, float height){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width=1;
  int cont=0;
  float angle=std::atan(float(radius)/height);
  for(float h=-height/2; h<height/2; h+=0.25){
    float radius_height=std::tan(angle)*((height/2)+h);
    for(float i=0; i<side_matrix_; i+=0.25){
      for(float j=0; j<side_matrix_; j+=0.25){
        if(std::sqrt(((i-side_matrix_/2)*(i-side_matrix_/2))+((j-side_matrix_/2)*(j-side_matrix_/2)))<=radius_height){
          cont++;
          cloud->height=cont;
          cloud->points.resize(cont);
          cloud->points[cont-1].x=i;
          cloud->points[cont-1].y=j;
          cloud->points[cont-1].z=side_matrix_/2+h;

        }

      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::generateSphere(float radius){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width=1;
  int cont=0;
  for (float i = 0; i < side_matrix_; i+=0.25) {
    for (float j = 0; j < side_matrix_; j+=0.25) {
      for (float k = 0; k < side_matrix_; k+=0.25) {
        if (std::sqrt(((i - side_matrix_/2) * (i - side_matrix_/2)) + ((j - side_matrix_/2) * (j - side_matrix_/2)) + ((k - side_matrix_/2) * (k - side_matrix_/2))) <= radius) {
          cont++;
          cloud->height = cont;
          cloud->points.resize(cont);
          cloud->points[cont - 1].x = i;
          cloud->points[cont - 1].y = j;
          cloud->points[cont - 1].z = k;
        }
      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::rotatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rot_x, float rot_y, float rot_z){
  Eigen::Affine3f rot=Eigen::Affine3f::Identity();
  rot.rotate(Eigen::AngleAxisf(rot_x, Eigen::Vector3f::UnitX()));
  rot.rotate(Eigen::AngleAxisf(rot_y, Eigen::Vector3f::UnitY()));
  rot.rotate(Eigen::AngleAxisf(rot_z, Eigen::Vector3f::UnitZ()));
  Eigen::Affine3f trans=Eigen::Affine3f::Identity();
  trans.translation() << -side_matrix_/2, -side_matrix_/2, -side_matrix_/2;
  Eigen::Affine3f trans2=Eigen::Affine3f::Identity();
  trans2.translation() << side_matrix_/2, side_matrix_/2, side_matrix_/2;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud(*cloud, *rotated_cloud, trans);
  pcl::transformPointCloud(*rotated_cloud, *rotated_cloud, rot);
  pcl::transformPointCloud(*rotated_cloud, *rotated_cloud, trans2);
  return rotated_cloud;

}
pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::translatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float trans_x, float trans_y, float trans_z){
  Eigen::Affine3f trans=Eigen::Affine3f::Identity();
  trans.translation() << trans_x, trans_y, trans_z;
  pcl::PointCloud<pcl::PointXYZ>::Ptr translated_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud(*cloud, *translated_cloud, trans);
  return translated_cloud;
}
std::vector< std::vector <std::vector< int > > > ObjectDatasetGenerator::getMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  std::vector< std::vector< std::vector< int > > > mat;
  for(int l=0; l<side_matrix_; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<side_matrix_; m++){
      std::vector<int> vec;
      for(int n=0; n<side_matrix_; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    mat.push_back(vec_vec);
  }
  for(int i=0; i<cloud->size(); i++){
    if(cloud->points[i].x>=side_matrix_ || cloud->points[i].x<0 || cloud->points[i].y>=side_matrix_ || cloud->points[i].y<0 || cloud->points[i].z>=side_matrix_ || cloud->points[i].z<0){
      std::cout<<"Error rotation, out of matrix: "<<cloud->points[i].x<<" "<<cloud->points[i].y<<" "<<cloud->points[i].z<<std::endl;
      mat.resize(0);
      return mat;
    }
    else
      mat[cloud->points[i].x][cloud->points[i].y][cloud->points[i].z]=1;
  }
  return mat;
}
std::vector< std::vector <std::vector< int > > > ObjectDatasetGenerator::getSideMatrix(std::vector< std::vector <std::vector< int > > > mat){
  std::vector< std::vector <std::vector< int > > > side_mat;
  for(int l=0; l<side_matrix_; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<side_matrix_; m++){
      std::vector<int> vec;
      for(int n=0; n<side_matrix_; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    side_mat.push_back(vec_vec);
  }


  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      bool found=false;
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          if(!found){
            side_mat[i][j][k]=1;
            found=true;
          }
        }
      }
    }
  }
  return side_mat;
}

std::vector< std::vector <std::vector< int > > > ObjectDatasetGenerator::getHollowMatrix(std::vector< std::vector <std::vector< int > > > mat){
  std::vector< std::vector <std::vector< int > > > hollow_mat;
  for(int l=0; l<side_matrix_; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<side_matrix_; m++){
      std::vector<int> vec;
      for(int n=0; n<side_matrix_; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    hollow_mat.push_back(vec_vec);
  }


  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          bool middle=true;
          for(int l=-1; l<=1; l++){
            for(int m=-1; m<=1; m++){
              for(int n=-1; n<=1; n++){
		if(i==0 || i==side_matrix_-1 || j==0 || j==side_matrix_-1 || k==0 || k==side_matrix_-1){
		  middle=false;	
		}
                else if(mat[i+l][j+m][k+m]==0){
                  middle=false;
                }
              }
            }
          }
          if(middle)
            hollow_mat[i][j][k]=0;
          else
            hollow_mat[i][j][k]=1;
        }
        else{
          hollow_mat[i][j][k]=0;
        }
      }
    }
  }
  return hollow_mat;
}
std::vector<int> ObjectDatasetGenerator::getDisplacement(std::vector< std::vector <std::vector< int > > > mat)
{
  int max_x=0, max_y=0, max_z=0, min_x=side_matrix_, min_y=side_matrix_, min_z=side_matrix_;
  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          if(i>max_x)
            max_x=i;
          if(i<min_x)
            min_x=i;
          if(j>max_y)
            max_y=j;
          if(j<min_y)
            min_y=j;
          if(k>max_z)
            max_z=k;
          if(k<min_z)
            min_z=k;
        }
      }
    }
  }
  int disp_x=side_matrix_/2-((max_x+min_x)/2);
  int disp_y=side_matrix_/2-((max_y+min_y)/2);
  //int disp_z=((max_z+min_z)/2)-25;
  int disp_z=-min_z;
  std::vector<int> disp;
  disp.push_back(disp_x);
  disp.push_back(disp_y);
  disp.push_back(disp_z);
  return disp;
}
std::vector< std::vector <std::vector< int > > > ObjectDatasetGenerator::moveMatrix(std::vector< std::vector <std::vector< int > > > mat, std::vector<int> disp){
  std::vector< std::vector <std::vector< int > > > moved_mat;
  for(int l=0; l<side_matrix_; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<side_matrix_; m++){
      std::vector<int> vec;
      for(int n=0; n<side_matrix_; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    moved_mat.push_back(vec_vec);
  }
  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          if(i+disp[0]>=side_matrix_ || i+disp[0]<0 || j+disp[1]>=side_matrix_ || j+disp[1]<0 || k+disp[2]>=side_matrix_ || k+disp[2]<0){
            std::cout<<"Error move matrix, out of matrix: "<<i+disp[0]<<" "<<j+disp[1]<<" "<<k+disp[2]<<std::endl;
            moved_mat.resize(0);
            return moved_mat;
          }
          else
            moved_mat[i+disp[0]][j+disp[1]][k+disp[2]]=1;
        }
      }
    }
  }
  return moved_mat;
}
int ObjectDatasetGenerator::visualizeMat(std::vector< std::vector <std::vector< int > > > mat, float r, float g, float b, int start=0){
  int cont=0;
  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          std::ostringstream name;
          name<<cont+start;
          viewer_->addCube(i, i+1, j, j+1, k, k+1, r, g, b, name.str());
          cont++;
        }
      }
    }
  }

  return cont+start;
}
void ObjectDatasetGenerator::visualizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string name){
  viewer_->addPointCloud<pcl::PointXYZ>(cloud, name);
  viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, name);
}
pcl::PointCloud<pcl::PointXYZ>::Ptr ObjectDatasetGenerator::getSidePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated, std::vector<std::vector<std::vector<int> > > side_mat){
  pcl::PointCloud<pcl::PointXYZ>::Ptr side_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  int cont=0;
  for(int i=0; i<cloud_rotated->points.size(); i++){
    if(side_mat[(int)cloud_rotated->points[i].x][(int)cloud_rotated->points[i].y][(int)cloud_rotated->points[i].z]==1){
      cont++;
      side_cloud->height=cont;
      side_cloud->points.resize(cont);
      side_cloud->points[cont-1].x=cloud_rotated->points[i].x;
      side_cloud->points[cont-1].y=cloud_rotated->points[i].y;
      side_cloud->points[cont-1].z=cloud_rotated->points[i].z;
    }
  }
  return side_cloud;
}

std::vector< std::vector< std::vector <std::vector< int > > > > ObjectDatasetGenerator::generateMats(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z, float camera_rot_x, float camera_rot_y, float camera_rot_z, float camera_trans_x, float camera_trans_y, float camera_trans_z){
  std::vector< std::vector< std::vector <std::vector< int > > > > mats;


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated=rotatePointCloud(cloud, rand_x, rand_y, rand_z);

  std::vector< std::vector <std::vector< int > > > mat=getMatrix(cloud_rotated);
  if(mat.size()==0){
    mats.resize(0);
    return mats;
  }

  std::vector< std::vector <std::vector< int > > > side_mat=getSideMatrix(mat);
  pcl::PointCloud<pcl::PointXYZ>::Ptr side_cloud=getSidePointCloud(cloud_rotated, side_mat);


  pcl::PointCloud<pcl::PointXYZ>::Ptr arm_cloud=rotatePointCloud(cloud_rotated, camera_rot_x, camera_rot_y, camera_rot_z);
  pcl::PointCloud<pcl::PointXYZ>::Ptr arm_side_cloud=rotatePointCloud(side_cloud, camera_rot_x, camera_rot_y, camera_rot_z);



  std::vector< std::vector <std::vector< int > > > mat_arm=getMatrix(translatePointCloud(arm_cloud, camera_trans_x, camera_trans_y, camera_trans_z));
  if(mat_arm.size()==0){
    mats.resize(0);
    return mats;
  }
  std::vector< std::vector <std::vector< int > > > side_mat_arm=getMatrix(translatePointCloud(arm_side_cloud, camera_trans_x, camera_trans_y, camera_trans_z));
  if(side_mat_arm.size()==0){
    mats.resize(0);
    return mats;
  }


  /*while(!viewer->wasStopped()){
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }*/




  mats.push_back(side_mat_arm);
  mats.push_back(mat_arm);
  return mats;
}

std::vector< std::vector< std::vector <std::vector< int > > > > ObjectDatasetGenerator::generateMatsNoCamera(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z){
  std::vector< std::vector< std::vector <std::vector< int > > > > mats;


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated=rotatePointCloud(cloud, rand_x, rand_y, rand_z);

  std::vector< std::vector <std::vector< int > > > mat=getMatrix(cloud_rotated);
  if(mat.size()==0){
    mats.resize(0);
    return mats;
  }

  std::vector< std::vector <std::vector< int > > > side_mat=getSideMatrix(mat);
  std::vector<int> displacement=getDisplacement(side_mat);





  mats.push_back(moveMatrix(side_mat, displacement));
  mats.push_back(moveMatrix(mat, displacement));
  return mats;
}



std::vector< std::vector< std::vector <std::vector< int > > > > ObjectDatasetGenerator::generateMatsNoCameraHollowObject(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z){
  std::vector< std::vector< std::vector <std::vector< int > > > > mats;


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated=rotatePointCloud(cloud, rand_x, rand_y, rand_z);

  std::vector< std::vector <std::vector< int > > > mat=getMatrix(cloud_rotated);
  if(mat.size()==0){
    mats.resize(0);
    return mats;
  }
  std::vector< std::vector <std::vector< int > > > hollow_mat=getHollowMatrix(mat);
  




  mats.push_back(hollow_mat);
  mats.push_back(mat);
  return mats;
}

void ObjectDatasetGenerator::writeMat(std::vector< std::vector <std::vector< int > > > mat, const std::string file_name){

  std::ofstream file(file_name.c_str());
  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        file<<mat[i][j][k]<<" ";
      }
    }
  }
  file.close();
}





void ObjectDatasetGenerator::generateDataset(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator){
  std::string training_validation;
  if(training){
    training_validation="/training";
    std::cout<<"training"<<std::endl;
  }
  else{
    training_validation="/validation";
    std::cout<<"validation"<<std::endl;

  }

  for(int i=0; i<num_objects/(cubes+cylinders+cones+spheres)/orientations; i++){
    for (int cube=0; cube<cubes; cube++){
      float x=(((std::rand()%100)/100.0)*x_max_)+x_min_;
      float y=(((std::rand()%100)/100.0)*y_max_)+y_min_;
      float z=(((std::rand()%100)/100.0)*z_max_)+z_min_;

      pcl::PointCloud<pcl::PointXYZ>::Ptr cube_cloud=generateCube(x,y,z);

      for(int orientation=0; orientation<orientations; orientation++){

        bool good=false;
        while(!good){
          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          float camera_rot_x=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_y=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_z=(std::rand()%1000)/1000.0*3.1415;

          float camera_trans_x=((std::rand()%100)/100.0*40)-20;
          float camera_trans_y=((std::rand()%100)/100.0*40)-20;
          float camera_trans_z=((std::rand()%100)/100.0*40)-20;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMats(cube_cloud, rand_x, rand_y, rand_z, camera_rot_x, camera_rot_y, camera_rot_z, camera_trans_x, camera_trans_y, camera_trans_z);
          if(mats.size()!=0){
            good=true;


            cont_++;
            if(training)
              general_<<cont_<<" "<<"cube trainig"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            else
              general_<<cont_<<" "<<"cube validation"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cube"<<std::endl;
          }
        }
      }

    }

    for (int cylinder=0; cylinder<cylinders; cylinder++){
      float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;
      float height=(((std::rand()%100)/100.0)*height_max_)+height_min_;

      pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud=generateCylinder(radius,height);
      for(int orientation=0; orientation<orientations; orientation++){
        bool good=false;
        while(!good){
          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          float camera_rot_x=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_y=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_z=(std::rand()%1000)/1000.0*3.1415;

          float camera_trans_x=((std::rand()%100)/100.0*40)-20;
          float camera_trans_y=((std::rand()%100)/100.0*40)-20;
          float camera_trans_z=((std::rand()%100)/100.0*40)-20;

          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMats(cylinder_cloud, rand_x, rand_y, rand_z, camera_rot_x, camera_rot_y, camera_rot_z, camera_trans_x, camera_trans_y, camera_trans_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"cylinder training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            else
              general_<<cont_<<" "<<"cylinder validation"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cylinder"<<std::endl;
          }
        }


      }

    }
    for (int cone=0; cone<cones; cone++){
      float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;
      float height=(((std::rand()%100)/100.0)*height_max_)+height_min_;


      pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cloud=generateCone(radius,height);
      for(int orientation=0; orientation<orientations; orientation++){
        bool good=false;
        while(!good){
          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          float camera_rot_x=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_y=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_z=(std::rand()%1000)/1000.0*3.1415;

          float camera_trans_x=((std::rand()%100)/100.0*40)-20;
          float camera_trans_y=((std::rand()%100)/100.0*40)-20;
          float camera_trans_z=((std::rand()%100)/100.0*40)-20;

          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMats(cone_cloud, rand_x, rand_y, rand_z, camera_rot_x, camera_rot_y, camera_rot_z, camera_trans_x, camera_trans_y, camera_trans_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"cone training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            else
              general_<<cont_<<" "<<"cone validation"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cone"<<std::endl;
          }
        }

      }
    }


    for (int sphere=0; sphere<spheres; sphere++){
      float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;

      pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_cloud=generateSphere(radius);
      for(int orientation=0; orientation<orientations; orientation++){
        bool good=false;
        while(!good){
          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          float camera_rot_x=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_y=(std::rand()%1000)/1000.0*3.1415;
          float camera_rot_z=(std::rand()%1000)/1000.0*3.1415;

          float camera_trans_x=((std::rand()%100)/100.0*40)-20;
          float camera_trans_y=((std::rand()%100)/100.0*40)-20;
          float camera_trans_z=((std::rand()%100)/100.0*40)-20;

          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMats(sphere_cloud, rand_x, rand_y, rand_z, camera_rot_x, camera_rot_y, camera_rot_z, camera_trans_x, camera_trans_y, camera_trans_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"sphere training"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            else
              general_<<cont_<<" "<<"sphere validation"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" sphere"<<std::endl;
          }
        }
      }
    }
  }

}

void ObjectDatasetGenerator::generateDatasetNoCamera(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator){



  std::string training_validation;
  if(training){
    training_validation="/training";
    std::cout<<"training"<<std::endl;
  }
  else{
    training_validation="/validation";
    std::cout<<"validation"<<std::endl;
  }

  std::ofstream types;
  std::cout<<(folder_+iterator+training_validation+"/types.txt").c_str()<<std::endl;
  types.open((folder_+iterator+training_validation+"/types.txt").c_str(), std::ofstream::out | std::ofstream::trunc);

  for(int i=0; i<num_objects/(cubes+cylinders+cones+spheres)/orientations; i++){



    for (int cube=0; cube<cubes; cube++){
      bool good=false;
      while(!good){
        float x=(((std::rand()%100)/100.0)*x_max_)+x_min_;
        float y=(((std::rand()%100)/100.0)*y_max_)+y_min_;
        float z=(((std::rand()%100)/100.0)*z_max_)+z_min_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cube_cloud=generateCube(x,y,z);

        for(int orientation=0; orientation<orientations; orientation++){


          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCamera(cube_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;


            cont_++;
            if(training)
              general_<<cont_<<" "<<"cube trainig"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"cube validation"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"cube"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cube"<<std::endl;
          }
        }
      }

    }

    for (int cylinder=0; cylinder<cylinders; cylinder++){
      bool good=false;
      while(!good){
        float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;
        float height=(((std::rand()%100)/100.0)*height_max_)+height_min_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud=generateCylinder(radius,height);
        for(int orientation=0; orientation<orientations; orientation++){

          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCamera(cylinder_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"cylinder training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"cylinder validation"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"cylinder"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cylinder"<<std::endl;
          }
        }


      }

    }
    for (int cone=0; cone<cones; cone++){
      bool good=false;
      while(!good){
        float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;
        float height=(((std::rand()%100)/100.0)*height_max_)+height_min_;


        pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cloud=generateCone(radius,height);
        for(int orientation=0; orientation<orientations; orientation++){


          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;



          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCamera(cone_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"cone training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"cone validation"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"cone"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cone"<<std::endl;
          }
        }

      }
    }


    for (int sphere=0; sphere<spheres; sphere++){
      bool good=false;
      while(!good){
        float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_cloud=generateSphere(radius);
        for(int orientation=0; orientation<orientations; orientation++){


          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCamera(sphere_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"sphere training"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"sphere validation"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"sphere"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" sphere"<<std::endl;
          }
        }
      }
    }
  }
  types.close();
}

void ObjectDatasetGenerator::generateDatasetHollowObject(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator){


  std::string training_validation;
  if(training){
    training_validation="/training";
    std::cout<<"training"<<std::endl;
  }
  else{
    training_validation="/validation";
    std::cout<<"validation"<<std::endl;
  }

  std::ofstream types;
  std::cout<<(folder_+iterator+training_validation+"/types.txt").c_str()<<std::endl;
  types.open((folder_+iterator+training_validation+"/types.txt").c_str(), std::ofstream::out | std::ofstream::trunc);

  for(int i=0; i<num_objects/(cubes+cylinders+cones+spheres)/orientations; i++){



    for (int cube=0; cube<cubes; cube++){
      bool good=false;
      while(!good){
        float x=(((std::rand()%100)/100.0)*x_max_)+x_min_;
        float y=(((std::rand()%100)/100.0)*y_max_)+y_min_;
        float z=(((std::rand()%100)/100.0)*z_max_)+z_min_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cube_cloud=generateCube(x,y,z);

        for(int orientation=0; orientation<orientations; orientation++){


          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCameraHollowObject(cube_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;


            cont_++;
            if(training)
              general_<<cont_<<" "<<"cube trainig"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"cube validation"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"cube"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cube"<<std::endl;
          }
        }
      }

    }

    for (int cylinder=0; cylinder<cylinders; cylinder++){
      bool good=false;
      while(!good){
        float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;
        float height=(((std::rand()%100)/100.0)*height_max_)+height_min_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud=generateCylinder(radius,height);
        for(int orientation=0; orientation<orientations; orientation++){

          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCameraHollowObject(cylinder_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"cylinder training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"cylinder validation"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"cylinder"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cylinder"<<std::endl;
          }
        }


      }

    }
    for (int cone=0; cone<cones; cone++){
      bool good=false;
      while(!good){
        float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;
        float height=(((std::rand()%100)/100.0)*height_max_)+height_min_;


        pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cloud=generateCone(radius,height);
        for(int orientation=0; orientation<orientations; orientation++){


          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;



          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCameraHollowObject(cone_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"cone training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"cone validation"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"cone"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" cone"<<std::endl;
          }
        }

      }
    }


    for (int sphere=0; sphere<spheres; sphere++){
      bool good=false;
      while(!good){
        float radius=(((std::rand()%100)/100.0)*radius_max_)+radius_min_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr sphere_cloud=generateSphere(radius);
        for(int orientation=0; orientation<orientations; orientation++){


          float rand_x=(std::rand()%1000)/1000.0*3.1415;
          float rand_y=(std::rand()%1000)/1000.0*3.1415;
          float rand_z=(std::rand()%1000)/1000.0*3.1415;


          std::vector< std::vector< std::vector <std::vector< int > > > > mats=generateMatsNoCameraHollowObject(sphere_cloud, rand_x, rand_y, rand_z);
          if(mats.size()!=0){
            good=true;
            cont_++;
            if(training)
              general_<<cont_<<" "<<"sphere training"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;
            else
              general_<<cont_<<" "<<"sphere validation"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            types<<cont_<<" "<<"sphere"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<std::endl;

            std::ostringstream id;
            id<<cont_;

            writeMat(mats[0], folder_+iterator+training_validation+"/side_objects/"+id.str()+".txt");
            writeMat(mats[1], folder_+iterator+training_validation+"/complete_objects/"+id.str()+".txt");
            std::cout<<cont_<<" sphere"<<std::endl;
          }
        }
      }
    }
  }
  types.close();
}

