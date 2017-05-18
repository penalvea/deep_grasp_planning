#include "deep_grasp_planning/object_dataset_generator.h"

int main(int argc, char** argv){
  int num_data_training=480;
  int num_data_validation=40;
  int orientations=1;
  int cubes=8;
  int cylinders=1;
  int cones=1;
  int spheres=0;

  std::string iterator="/first";
  ObjectDatasetGenerator dataset("/home/penalvea/dataset/geometrics/general.txt", "/home/penalvea/dataset/geometrics", 30);
  while(!std::ifstream("/home/penalvea/dataset/stop")){
    while(!std::ifstream("/home/penalvea/dataset/first") && !std::ifstream("/home/penalvea/dataset/stop")){
      sleep(10);
    }
     std::remove("/home/penalvea/dataset/first");
    if(!std::ifstream("/home/penalvea/dataset/stop")){


      std::string del;
      del="rm -r /home/penalvea/dataset/geometrics/first/training/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset/geometrics/first/training/side_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset/geometrics/first/validation/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset/geometrics/first/validation/side_objects/*";
      system(del.c_str());
      dataset.generateDatasetNoCamera(num_data_training, cubes, cylinders, cones, spheres, orientations, true, iterator);
      dataset.generateDatasetNoCamera(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, iterator);

      std::ofstream outfile ("/home/penalvea/dataset/first_ready");
      outfile.close();
    }
    while(!std::ifstream("/home/penalvea/dataset/second") && !std::ifstream("/home/penalvea/dataset/stop")){
      sleep(10);
    }
    std::remove("/home/penalvea/dataset/second");

    if(!std::ifstream("/home/penalvea/dataset/stop")){
      std::string del;
      del="rm -r /home/penalvea/dataset/geometrics/second/training/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset/geometrics/second/training/side_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset/geometrics/second/validation/complete_objects/*";
      system(del.c_str());
      del="rm -r /home/penalvea/dataset/geometrics/second/validation/side_objects/*";
      system(del.c_str());
      dataset.generateDatasetNoCamera(num_data_training, cubes, cylinders, cones, spheres, orientations, true, "/second");
      dataset.generateDatasetNoCamera(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, "/second");

      std::ofstream outfile2 ("/home/penalvea/dataset/second_ready");
      outfile2.close();
    }
  }



  return 0;
}
