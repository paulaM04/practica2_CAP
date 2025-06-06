# Especificar versión mínima de CMake
# CMake 3.20 es la versión más reciente que viene incluida con
# Visual Studio 2019
cmake_minimum_required(VERSION "3.20")

# Declaramos el proyecto de CMake
# En caso de que no haga falta alguno de los lenguajes listados,
# se puede deshabilitar simplemente eliminándolo de la lista
# https://cmake.org/cmake/help/latest/command/project.html
project(RayTracingCPU LANGUAGES "CXX")

#################################################
# Importación de dependencias
#################################################

# Buscar soporte de OpenMP para C++ como obligatorio
# Si no hace falta usar OpenMP, esta línea se puede comentar
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindOpenMP.html
find_package("OpenMP" REQUIRED COMPONENTS "CXX")

# Buscar una implementación de la Message Parsing Interface como obligatoria
# Si no hace falta usar MPI, esta línea se puede comentar
# https://cmake.org/cmake/help/latest/command/find_package.html
# https://cmake.org/cmake/help/latest/module/FindMPI.html
find_package("MPI" REQUIRED COMPONENTS "CXX")

#################################################
# Descripción del ejecutable a crear
#################################################

# Declarar ejecutable
# https://cmake.org/cmake/help/latest/command/add_executable.html
add_executable(raytracingcpu
    Camera.hpp
    CollisionData.hpp
    Crystalline.cpp
    Crystalline.hpp
    Diffuse.hpp
    Main.cpp
    Material.hpp
    Metallic.cpp
    Metallic.hpp
    Object.hpp
    Random.cpp
    Random.hpp
    Ray.hpp
    Scene.cpp
    Scene.hpp
    Shape.hpp
    Sphere.cpp
    Sphere.hpp
    Utils.cpp
    Utils.hpp
    Vec3.cpp
    Vec3.hpp
)

#[[
    Especificar el nombre del ejecutable compilado

    Por defecto, CMake nombra los ejecutables y las librerías que genera con el mismo
    nombre que el objetivo (target) a partir del cual se han generado. Sin embargo,
    la propiedad de objetivo "OUTPUT_NAME" se puede configurar para darle al ejecutable
    o librería generado un nombre distinto al del objetivo a partir del cual se ha generado.
]]
set_target_properties(raytracingcpu PROPERTIES OUTPUT_NAME "raytracingcpu")

# Enlazar el ejecutable contra las librerías necesarias
target_link_libraries(raytracingcpu PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)

# Declarar reglas de instalación
# Esto sirve para decirle a CMake cómo debe exportar los archivos compilados
# en el caso de que así se desee
# https://cmake.org/cmake/help/latest/command/install.html
# https://cmake.org/cmake/help/latest/manual/cmake.1.html#install-a-project
install(TARGETS raytracingcpu)
