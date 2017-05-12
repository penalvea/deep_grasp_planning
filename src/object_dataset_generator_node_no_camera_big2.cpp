#include "deep_grasp_planning/object_dataset_generator.h"

int main(int argc, char** argv){
  int num_data_training=240;
  int num_data_validation=8;
  int orientations=1;
  int cubes=3;
  int cylinders=2;
  int cones=2;
  int spheres=1;

  int workspace=60;

  std::string iterator="/second";
  ObjectDatasetGenerator dataset("/home/penalvea/dataset2/geometrics/general.txt", "/home/penalvea/dataset2/geometrics", workspace);
  dataset.change_sizes(40, 8, 40, 8, 30, 8, 40, 8, 12, 4);

  while(!std::ifstream("/home/penalvea/dataset2/stop")){
    while(!std::ifstream("/home/penalvea/dataset2/second") && !std::ifstream("/home/penalvea/dataset2/stop")){
      sleep(10);
    }
    std::remove("/home/penalvea/dataset2/second");
    if(!std::ifstream("/home/penalvea/dataset2/stop")){


      std::string del;
      del="rm -r /home/penalvea/dataset2/geometrics/second/training/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset2/geometrics/second/training/side_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset2/geometrics/second/validation/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset2/geometrics/second/validation/side_objects/*";
      system(del.c_str());
      dataset.generateDatasetNoCamera(num_data_training, cubes, cylinders, cones, spheres, orientations, true, iterator);
      dataset.generateDatasetNoCamera(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, iterator);

      std::ofstream outfile ("/home/penalvea/dataset2/second_ready");
      outfile.close();
    }

  }


  return 0;
}