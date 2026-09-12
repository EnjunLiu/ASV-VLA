cmake_minimum_required(VERSION 3.8)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ASVState.msg"
  "msg/EntityState.msg"
  "msg/EntityStateArray.msg"
  "msg/CameraFrame.msg"
  "msg/DesiredDisplacement.msg"
  "msg/Input.msg"
  "msg/Output.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
