#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>

int cont=0;

pcl::PointCloud<pcl::PointXYZ>::Ptr generateCube(float x, float y, float z){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->height=1;
  int cont=0;
  for(float i=-x/2; i<x/2; i+=0.25){
    for(float j=-y/2; j<y/2; j+=0.25){
      for(float k=-z/2; k<z/2; k+=0.25){
        cont++;
        cloud->width=cont;
        cloud->points.resize(cont);
        cloud->points[cont-1].x=25+i;
        cloud->points[cont-1].y=25+j;
        cloud->points[cont-1].z=25+k;
      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr generateCylinder(float radius, float height){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width=1;
  int cont=0;
  for(float h=-height/2; h<height/2; h+=0.25){
    for(float i=0; i<50; i+=0.25){
      for(float j=0; j<50; j+=0.25){
        if(std::sqrt(((i-25)*(i-25))+((j-25)*(j-25)))<=radius){
          cont++;
          cloud->height=cont;
          cloud->points.resize(cont);
          cloud->points[cont-1].x=i;
          cloud->points[cont-1].y=j;
          cloud->points[cont-1].z=25+h;
        }

      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr generateCone(float radius, float height){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width=1;
  int cont=0;
  float angle=std::atan(float(radius)/height);
  for(float h=-height/2; h<height/2; h+=0.25){
    float radius_height=std::tan(angle)*((height/2)+h);
    for(float i=0; i<50; i+=0.25){
      for(float j=0; j<50; j+=0.25){
        if(std::sqrt(((i-25)*(i-25))+((j-25)*(j-25)))<=radius_height){
          cont++;
          cloud->height=cont;
          cloud->points.resize(cont);
          cloud->points[cont-1].x=i;
          cloud->points[cont-1].y=j;
          cloud->points[cont-1].z=25+h;

        }

      }
    }
  }
  return cloud;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSphere(float radius){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width=1;
  int cont=0;
  for (float i = 0; i < 50; i+=0.25) {
    for (float j = 0; j < 50; j+=0.25) {
      for (float k = 0; k < 50; k+=0.25) {
        if (std::sqrt(((i - 25) * (i - 25)) + ((j - 25) * (j - 25)) + ((k - 25) * (k - 25))) <= radius) {
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
pcl::PointCloud<pcl::PointXYZ>::Ptr rotatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rot_x, float rot_y, float rot_z){
  Eigen::Affine3f rot=Eigen::Affine3f::Identity();
  rot.rotate(Eigen::AngleAxisf(rot_x, Eigen::Vector3f::UnitX()));
  rot.rotate(Eigen::AngleAxisf(rot_y, Eigen::Vector3f::UnitY()));
  rot.rotate(Eigen::AngleAxisf(rot_z, Eigen::Vector3f::UnitZ()));
  Eigen::Affine3f trans=Eigen::Affine3f::Identity();
  trans.translation() << -25, -25, -25;
  Eigen::Affine3f trans2=Eigen::Affine3f::Identity();
  trans2.translation() << 25, 25, 25;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud(*cloud, *rotated_cloud, trans);
  pcl::transformPointCloud(*rotated_cloud, *rotated_cloud, rot);
  pcl::transformPointCloud(*rotated_cloud, *rotated_cloud, trans2);
  return rotated_cloud;

}
pcl::PointCloud<pcl::PointXYZ>::Ptr translatePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float trans_x, float trans_y, float trans_z){
  Eigen::Affine3f trans=Eigen::Affine3f::Identity();
  trans.translation() << trans_x, trans_y, trans_z;
  pcl::PointCloud<pcl::PointXYZ>::Ptr translated_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud(*cloud, *translated_cloud, trans);
  return translated_cloud;
}
std::vector< std::vector <std::vector< int > > > getMatrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  std::vector< std::vector< std::vector< int > > > mat;
  for(int l=0; l<50; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<50; m++){
      std::vector<int> vec;
      for(int n=0; n<50; n++){
        vec.push_back(0);
      }
      vec_vec.push_back(vec);
    }
    mat.push_back(vec_vec);
  }
  for(int i=0; i<cloud->size(); i++){
    if(cloud->points[i].x>=50 || cloud->points[i].x<0 || cloud->points[i].y>=50 || cloud->points[i].y<0 || cloud->points[i].z>=50 || cloud->points[i].z<0){
      std::cout<<"Error rotation, out of matrix: "<<cloud->points[i].x<<" "<<cloud->points[i].y<<" "<<cloud->points[i].z<<std::endl;
      mat.resize(0);
      return mat;
    }
    else
      mat[cloud->points[i].x][cloud->points[i].y][cloud->points[i].z]=1;
  }
  return mat;
}
std::vector< std::vector <std::vector< int > > > getSideMatrix(std::vector< std::vector <std::vector< int > > > mat){
  std::vector< std::vector <std::vector< int > > > side_mat;
  for(int l=0; l<50; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<50; m++){
      std::vector<int> vec;
      for(int n=0; n<50; n++){
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
std::vector<int> getDisplacement(std::vector< std::vector <std::vector< int > > > mat)
{
  int max_x=0, max_y=0, max_z=0, min_x=50, min_y=50, min_z=50;
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
  int disp_x=25-((max_x+min_x)/2);
  int disp_y=25-((max_y+min_y)/2);
  //int disp_z=((max_z+min_z)/2)-25;
  int disp_z=-min_z;
  std::vector<int> disp;
  disp.push_back(disp_x);
  disp.push_back(disp_y);
  disp.push_back(disp_z);
  return disp;
}
std::vector< std::vector <std::vector< int > > > moveMatrix(std::vector< std::vector <std::vector< int > > > mat, std::vector<int> disp){
  std::vector< std::vector <std::vector< int > > > moved_mat;
  for(int l=0; l<50; l++){
    std::vector<std::vector<int> > vec_vec;
    for(int m=0; m<50; m++){
      std::vector<int> vec;
      for(int n=0; n<50; n++){
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
          if(i+disp[0]>49 || i+disp[0]<0 || j+disp[1]>49 || j+disp[1]<0 || k+disp[2]>49 || k+disp[2]<0){
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
int visualizeMat(pcl::visualization::PCLVisualizer::Ptr viewer, std::vector< std::vector <std::vector< int > > > mat, float r, float g, float b, int start=0){
  int cont=0;
  for(int i=0; i<mat.size(); i++){
    for(int j=0; j<mat[0].size(); j++){
      for(int k=0; k<mat[0][0].size(); k++){
        if(mat[i][j][k]==1){
          std::ostringstream name;
          name<<cont+start;
          viewer->addCube(i, i+1, j, j+1, k, k+1, r, g, b, name.str());
          cont++;
        }
      }
    }
  }

  return cont+start;
}
void visualizePointCloud(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string name){
  viewer->addPointCloud<pcl::PointXYZ>(cloud, name);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, name);
}
pcl::PointCloud<pcl::PointXYZ>::Ptr getSidePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotated, std::vector<std::vector<std::vector<int> > > side_mat){
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

std::vector< std::vector< std::vector <std::vector< int > > > > generateMats(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float rand_x, float rand_y, float rand_z, float camera_rot_x, float camera_rot_y, float camera_rot_z, float camera_trans_x, float camera_trans_y, float camera_trans_z){
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
//visualizePointCloud(viewer, translatePointCloud(arm_side_cloud, camera_trans_x, camera_trans_y, camera_trans_z), "aaa");
  //visualizeMat(viewer, mat_arm, 1.0, 0.0, 0.0);
  //visualizeMat(viewer, side_mat_arm, 0.0, 1.0, 0.0, 10000000);



  /*while(!viewer->wasStopped()){
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }*/
  //std::vector<int> displacement=getDisplacement(side_mat);
  //std::vector< std::vector <std::vector< int > > > disp_side_mat=moveMatrix(side_mat, displacement);
  //std::vector< std::vector <std::vector< int > > > disp_mat=moveMatrix(mat, displacement);


  //mats.push_back(disp_mat);
  //mats.push_back(disp_side_mat);
  mats.push_back(mat_arm);
  mats.push_back(side_mat_arm);
  return mats;
}

void writeMat(std::vector< std::vector <std::vector< int > > > mat, const std::string file_name){
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

void generateDataset(int num_objects, int cubes, int cylinders, int cones, int spheres, int orientations, bool training, std::string iterator, std::ofstream general, std::string folder){
  std::string training_validation;
  if(training){
    training_validation="/training";
  }
  else{
    training_validation="/validation";
  }

  for(int i=0; i<num_objects/(cubes+cylinders+cones+spheres)/orientations; i++){
    for (int cube=0; cube<cubes; cube++){
      float x=(((std::rand()%100)/100.0)*16)+4;
      float y=(((std::rand()%100)/100.0)*16)+4;
      float z=(((std::rand()%100)/100.0)*11)+4;

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


            cont++;
            general<<cont<<" "<<"cube trainig"<<" "<<x<<" "<<y<<" "<<z<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            std::ostringstream id;
            id<<cont;

            writeMat(mats[0], folder+iterator+training_validation+"/side_objects/"+id.str()+".mat");
            writeMat(mats[1], folder+iterator+training_validation+"/complete_objects/"+id.str()+".mat");
            std::cout<<cont<<" cube"<<std::endl;
          }
        }
      }

    }

    for (int cylinder=0; cylinder<cylinders; cylinder++){
      float radius=(((std::rand()%100)/100.0)*6)+2;
      float height=(((std::rand()%100)/100.0)*16)+4;

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
            cont++;
            general<<cont<<" "<<"cylinder training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            std::ostringstream id;
            id<<cont;

            writeMat(mats[0], folder+iterator+training_validation+"/side_objects/"+id.str()+".mat");
            writeMat(mats[1], folder+iterator+training_validation+"/complete_objects/"+id.str()+".mat");
            std::cout<<cont<<" cylinder"<<std::endl;
          }
        }


      }

    }
    for (int cone=0; cone<cones; cone++){
      float radius=(((std::rand()%100)/100.0)*6)+2;
      float height=(((std::rand()%100)/100.0)*16)+4;


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
            cont++;
            general<<cont<<" "<<"cone training"<<" "<<radius<<" "<<height<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            std::ostringstream id;
            id<<cont;

            writeMat(mats[0], folder+iterator+training_validation+"/side_objects/"+id.str()+".mat");
            writeMat(mats[1], folder+iterator+training_validation+"/complete_objects/"+id.str()+".mat");
            std::cout<<cont<<" cone"<<std::endl;
          }
        }

      }
    }


    for (int sphere=0; sphere<spheres; sphere++){
      float radius=(((std::rand()%100)/100.0)*6)+2;

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
            cont++;
            general<<cont<<" "<<"sphere training"<<" "<<radius<<" "<<"-1"<<" "<<"-1"<<" "<<rand_x<<" "<<rand_y<<" "<<rand_z<<" "<<camera_rot_x<<" "<<camera_rot_y<<" "<<camera_rot_z<<" "<<camera_trans_x<<" "<<camera_trans_y<<" "<<camera_trans_z<<std::endl;
            std::ostringstream id;
            id<<cont;

            writeMat(mats[0], folder+iterator+training_validation+"/side_objects/"+id.str()+".mat");
            writeMat(mats[1], folder+iterator+training_validation+"/complete_objects/"+id.str()+".mat");
            std::cout<<cont<<" sphere"<<std::endl;
          }
        }
      }
    }
  }

}

int main(){
  int num_data_training=40;
  int num_data_validation=10;
  int orientations=5;
  int cubes=3;
  int cylinders=2;
  int cones=2;
  int spheres=1;

  std::string iterator="/first";


  std::ofstream general("/home/penalvea/dataset/geometrics/general.txt");
  general<<"id shape training/validation x/radius y/height z rot_x rot_y rot_z camera_trans_x camera_trans_y camera_trans_z camera_rot_x camera_rot_y camera_rot_z\n";
  std::string folder="/home/penalvea/dataset/geometrics";
  std::srand(std::time(0));


  //pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  //viewer->setBackgroundColor(0,0,0);
  //viewer->addCoordinateSystem(50.0);
  //viewer->initCameraParameters();

  generateDataset(num_data_training, cubes, cylinders, cones, spheres, orientations, true, iterator, general, folder);
  generateDataset(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, iterator, general, folder);





  return 0;
}
