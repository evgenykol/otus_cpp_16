cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED ENV{TRAVIS_BUILD_NUMBER})
    message("TRAVIS_BUILD_NUMBER environment variable set to default")
    set(ENV{TRAVIS_BUILD_NUMBER} "1")
    set(BOOST_LIBRARYDIR "/opt/local/include/" )
endif()

project(realty VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})

configure_file(version.h.in version.h)
find_package(dlib REQUIRED)
#find_package(Boost COMPONENTS string REQUIRED)

add_executable(rclst rclst.cpp)
add_executable(rclss rclss.cpp)

add_library(rlib rlib.h rlib.cpp)

set_target_properties(rclst rclss rlib
        PROPERTIES
                CXX_STANDARD 14
                CXX_STANDARD_REQUIRED ON
                CMAKE_CXX_EXTENSIONS OFF
                COMPILE_OPTIONS -Wpedantic -Wall -Wextra
)

set_target_properties(rlib
    PROPERTIES
        #COMPILE_DEFINITIONS BOOST_TEST_DYN_LINK
        INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR}
)

target_link_libraries(rclst rlib
                        dlib::dlib
                        ${Boost_LIBRARIES}
)

target_link_libraries(rclss rlib
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
