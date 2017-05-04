#include "deep_grasp_planning/object_dataset_generator.h"

int main(int argc, char** argv){
  int num_data_training=40;
  int num_data_validation=40;
  int orientations=5;
  int cubes=3;
  int cylinders=2;
  int cones=2;
  int spheres=1;

  std::string iterator="/first";


  ObjectDatasetGenerator dataset("/home/penalvea/dataset/geometrics/general.txt", "/home/penalvea/dataset/geometrics", 50);
  std::string del;
  del="rm -r /home/penalvea/dataset/geometrics/first/training/complete_objects/*";
  system(del.c_str());
  del="rm -r /home/penalvea/dataset/geometrics/first/training/side_objects/*";
  system(del.c_str());
  del="rm -r /home/penalvea/dataset/geometrics/first/validation/complete_objects/*";
  system(del.c_str());
  del="rm -r /home/penalvea/dataset/geometrics/first/validation/side_objects/*";
  system(del.c_str());
  dataset.generateDataset(num_data_training, cubes, cylinders, cones, spheres, orientations, true, iterator);
  dataset.generateDataset(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, iterator);




  del="rm -r /home/penalvea/dataset/geometrics/second/training/complete_objects/*";
  system(del.c_str());
  del="rm -r /home/penalvea/dataset/geometrics/second/training/side_objects/*";
  system(del.c_str());
  del="rm -r /home/penalvea/dataset/geometrics/second/validation/complete_objects/*";
  system(del.c_str());
  del="rm -r /home/penalvea/dataset/geometrics/second/validation/side_objects/*";
  system(del.c_str());
  dataset.generateDataset(num_data_training, cubes, cylinders, cones, spheres, orientations, true, "/second");
  dataset.generateDataset(num_data_validation, cubes, cylinders, cones, spheres, orientations, false, "/second");





  return 0;
}
