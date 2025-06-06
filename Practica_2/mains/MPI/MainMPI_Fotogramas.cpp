#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
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

void rayTracingCPU(unsigned char* img, int w, int h, int ns = 10, int px = 0, int py = 0, int pw = -1, int ph = -1) {
    if (pw == -1) pw = w;
    if (ph == -1) ph = h;
    int patch_w = pw - px;
    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    for (int j = 0; j < (ph - py); j++) {
        for (int i = 0; i < (pw - px); i++) {

            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++) {
                float u = float(i + px + RayTracingCPU::random()) / float(w);
                float v = float(j + py + RayTracingCPU::random()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            img[(j * patch_w + i) * 3 + 2] = char(255.99 * col[0]);
            img[(j * patch_w + i) * 3 + 1] = char(255.99 * col[1]);
            img[(j * patch_w + i) * 3 + 0] = char(255.99 * col[2]);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ns = 10;
    int sizes[][2] = {
        {64, 64},
        {256, 256},
        {512, 512}
    };
    int num_resolutions = sizeof(sizes) / sizeof(sizes[0]);
    int num_frames = 3;

    for (int s = 0; s < num_resolutions; s++) {
        int w = sizes[s][0];
        int h = sizes[s][1];
        int img_size = sizeof(unsigned char) * w * h * 3;

        unsigned char* my_image = (unsigned char*)calloc(img_size, 1);

        std::clock_t start = std::clock();

        for (int frame = rank; frame < num_frames; frame += size) {
            rayTracingCPU(my_image, w, h, ns, 0, 0, w, h);
            break;
        }

        if (rank == 0) {
            unsigned char* final_image = (unsigned char*)calloc(img_size, 1);
            if (rank == 0) {
                memcpy(final_image, my_image, img_size);
            }
            MPI_Bcast(final_image, img_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

            char filename[128];
            sprintf(filename, "img_final_%dx%d_MPI.bmp", w, h);
            writeBMP(filename, final_image, w, h);
            free(final_image);
        } else {
            MPI_Bcast(my_image, img_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }

        std::clock_t end = std::clock();
        double duration = double(end - start) / CLOCKS_PER_SEC;

        if (rank == 0) {
            printf("Tiempo de CPU para %dx%d con %d procesos: %.3f segundos.\n",
               w, h, size, duration);

        }

        free(my_image);
    }

    MPI_Finalize();
    return 0;
}