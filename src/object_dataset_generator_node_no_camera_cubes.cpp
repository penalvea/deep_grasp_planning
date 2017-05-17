#include "deep_grasp_planning/object_dataset_generator.h"

int main(int argc, char** argv){
  int num_data_training=10000;
  int num_data_validation=100;
  int orientations=1;
  int cubes=100;
  int cylinders=0;
  int cones=0;
  int spheres=0;

  int workspace=30;

  std::string iterator="/first";
  ObjectDatasetGenerator dataset("/home/penalvea/cubes/general.txt", "/home/penalvea/cubes/geometrics", workspace);
  //dataset.change_sizes(40, 8, 40, 8, 30, 8, 40, 8, 12, 4);

  /*while(!std::ifstream("/home/penalvea/cylinders/stop")){
    while(!std::ifstream("/home/penalvea/cylinders/first") && !std::ifstream("/home/penalvea/cylinders/stop")){
      sleep(10);
    }
    std::remove("/home/penalvea/cylinders/first");
    if(!std::ifstream("/home/penalvea/cylinders/stop")){*/


      std::string del;
      del="rm -r /home/penalvea/cubes/first/training/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/cubes/first/training/side_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/cubes/first/validation/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/cubes/first/validation/side_objects/*";
      system(del.c_str());
      dataset.generateDatasetNoCamera(num_data_training, cubes, cylinders, cones, spheres, orientations, true, iterator);
      dataset.generateDatasetNoCamera(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, iterator);

      /*std::ofstream outfile ("/home/penalvea/cylinders/first_ready");
      outfile.close();
    }

  }*/


  return 0;
}
