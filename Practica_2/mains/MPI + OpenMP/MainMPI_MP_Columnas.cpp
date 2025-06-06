#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <sstream>
#include <mpi.h>
#include "Camera.hpp"
#include "Object.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "Diffuse.hpp"
#include "Metallic.hpp"
#include "Crystalline.hpp"
#include <omp.h>
#include "Random.hpp"
#include "Utils.hpp"

Scene randomScene() {
    int n = 500;
    Scene list;
    list.add(new Object(
        new Sphere(Vec3(0, -1000, 0), 1000),
        new Diffuse(Vec3(0.5, 0.5, 0.5))
    ));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RayTracingCPU::random();
            Vec3 center(a + 0.9f * RayTracingCPU::random(), 0.2f, b + 0.9f * RayTracingCPU::random());
            if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.8f) {  // diffuse
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Diffuse(Vec3(RayTracingCPU::random() * RayTracingCPU::random(),
                            RayTracingCPU::random() * RayTracingCPU::random(),
                            RayTracingCPU::random() * RayTracingCPU::random()))
                    ));
                }
                else if (choose_mat < 0.95f) { // metal
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Metallic(Vec3(0.5f * (1 + RayTracingCPU::random()),
                            0.5f * (1 + RayTracingCPU::random()),
                            0.5f * (1 + RayTracingCPU::random())),
                            0.5f * RayTracingCPU::random())
                    ));
                }
                else {  // glass
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Crystalline(1.5f)
                    ));
                }
            }
        }
    }

    list.add(new Object(
        new Sphere(Vec3(0, 1, 0), 1.0),
        new Crystalline(1.5f)
    ));
    list.add(new Object(
        new Sphere(Vec3(-4, 1, 0), 1.0f),
        new Diffuse(Vec3(0.4f, 0.2f, 0.1f))
    ));
    list.add(new Object(
        new Sphere(Vec3(4, 1, 0), 1.0f),
        new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)
    ));

    return list;
}


void rayTracingCPU(unsigned char* img, int w, int h, int ns = 10) {
    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + RayTracingCPU::random()) / float(w);
                float v = float(j + RayTracingCPU::random()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            int index = (j * w + i) * 3;
            img[index + 2] = static_cast<unsigned char>(255.99 * col[0]);
            img[index + 1] = static_cast<unsigned char>(255.99 * col[1]);
            img[index + 0] = static_cast<unsigned char>(255.99 * col[2]);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 3) {
        if (rank == 0) {
            printf("Uso: %s <width> <height>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int w = atoi(argv[1]);
    int h = atoi(argv[2]);
    int ns = 10;

    size_t image_size = w * h * 3 * sizeof(unsigned char);
    unsigned char* data = (unsigned char*)calloc(image_size, 1);

    double start_time = MPI_Wtime();
    rayTracingCPU(data, w, h, ns);
    double end_time = MPI_Wtime();

    std::ostringstream filename;
    filename << "frame_rank" << rank << ".bmp";
    writeBMP(filename.str().c_str(), data, w, h);

    if (rank == 0) {
        printf("Tiempo de CPU para %dx%d con %d procesos: %.3f segundos\n", w, h, num_procs, end_time - start_time);
    }

    free(data);
    MPI_Finalize();
    return 0;
}


