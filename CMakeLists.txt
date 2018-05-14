cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED ENV{TRAVIS_BUILD_NUMBER})
    message("TRAVIS_BUILD_NUMBER environment variable set to default")
    set(ENV{TRAVIS_BUILD_NUMBER} "1")
endif()

project(realty VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})

configure_file(version.h.in version.h)
find_package(dlib REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

add_executable(rclst rclst.cpp)
add_executable(rclss rclss.cpp)

set_target_properties(rclst rclss
        PROPERTIES
                CXX_STANDARD 14
                CXX_STANDARD_REQUIRED ON
                CMAKE_CXX_EXTENSIONS OFF
                COMPILE_OPTIONS -Wpedantic -Wall -Wextra
)

target_link_libraries(rclst
                        dlib::dlib
                        ${Boost_LIBRARIES}
)

target_link_libraries(rclss
                        dlib::dlib
                        ${Boost_LIBRARIES}
)

install(TARGETS rclst rclss RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT edkolotilov@gmail.com)

include (CPack)
