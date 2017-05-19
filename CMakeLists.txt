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

add_executable(object_dataset_generator_no_camera src/object_dataset_generator_node_no_camera.cpp)
target_link_libraries(object_dataset_generator_no_camera ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_big src/object_dataset_generator_node_no_camera_big.cpp)
target_link_libraries(object_dataset_generator_no_camera_big ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_big1 src/object_dataset_generator_node_no_camera_big1.cpp)
target_link_libraries(object_dataset_generator_no_camera_big1 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_big2 src/object_dataset_generator_node_no_camera_big2.cpp)
target_link_libraries(object_dataset_generator_no_camera_big2 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_big3 src/object_dataset_generator_node_no_camera_big3.cpp)
target_link_libraries(object_dataset_generator_no_camera_big3 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_big4 src/object_dataset_generator_node_no_camera_big4.cpp)
target_link_libraries(object_dataset_generator_no_camera_big4 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_big5 src/object_dataset_generator_node_no_camera_big5.cpp)
target_link_libraries(object_dataset_generator_no_camera_big5 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


add_executable(object_dataset_generator_no_camera_big6 src/object_dataset_generator_node_no_camera_big6.cpp)
target_link_libraries(object_dataset_generator_no_camera_big6 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


add_executable(object_dataset_generator_no_camera_big7 src/object_dataset_generator_node_no_camera_big7.cpp)
target_link_libraries(object_dataset_generator_no_camera_big7 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


add_executable(object_dataset_generator_no_camera_big8 src/object_dataset_generator_node_no_camera_big8.cpp)
target_link_libraries(object_dataset_generator_no_camera_big8 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


add_executable(object_dataset_generator_no_camera_big9 src/object_dataset_generator_node_no_camera_big9.cpp)
target_link_libraries(object_dataset_generator_no_camera_big9 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


add_executable(object_dataset_generator_no_camera_big10 src/object_dataset_generator_node_no_camera_big10.cpp)
target_link_libraries(object_dataset_generator_no_camera_big10 ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_cylinders src/object_dataset_generator_node_no_camera_cylinders.cpp)
target_link_libraries(object_dataset_generator_no_camera_cylinders ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_cubes src/object_dataset_generator_node_no_camera_cubes.cpp)
target_link_libraries(object_dataset_generator_no_camera_cubes ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


add_executable(object_dataset_generator_no_camera_cones src/object_dataset_generator_node_no_camera_cones.cpp)
target_link_libraries(object_dataset_generator_no_camera_cones ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_spheres src/object_dataset_generator_node_no_camera_spheres.cpp)
target_link_libraries(object_dataset_generator_no_camera_spheres ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(object_dataset_generator_no_camera_hollows src/object_dataset_generator_node_no_camera_hollows.cpp)
target_link_libraries(object_dataset_generator_no_camera_hollows ${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(read_mat_from_file src/read_mat_from_file.cpp)
target_link_libraries(read_mat_from_file ${catkin_LIBRARIES} ${PCL_LIBRARIES})



