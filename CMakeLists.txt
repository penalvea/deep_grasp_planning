cmake_minimum_required(VERSION 2.8.3)
project(deep_grasp_planning)


find_package(catkin REQUIRED COMPONENTS
  roscpp
)
find_package(PCL REQUIRED)






catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp
  DEPENDS
  	PCL
)



include_directories(
  ${PROJECT_SOURCE_DIR}/include
   ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${catkin_LIBRARY_DIRS})
add_library(${PROJECT_NAME} src/object_dataset_generator.cpp)



add_executable(generate_models src/generate_models.cpp)
target_link_libraries(generate_models ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(rotate_pointcloud src/rotate_pointcloud.cpp)
target_link_libraries(rotate_pointcloud ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator src/object_dataset_generator_node.cpp)
target_link_libraries(object_dataset_generator ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(generate_dataset src/generate_dataset.cpp)
#target_link_libraries(generate_dataset ${catkin_LIBRARIES} ${PCL_LIBRARIES})
