#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>

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

    const int tile_size = 32;

    int tile_rows = (ph - py + tile_size - 1) / tile_size;
    int tile_cols = (pw - px + tile_size - 1) / tile_size;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int tile_j = 0; tile_j < tile_rows; tile_j++) {
        for (int tile_i = 0; tile_i < tile_cols; tile_i++) {
            int start_i = tile_i * tile_size;
            int end_i = std::min(start_i + tile_size, pw - px);

            int start_j = tile_j * tile_size;
            int end_j = std::min(start_j + tile_size, ph - py);

            for (int j = start_j; j < end_j; j++) {
                for (int i = start_i; i < end_i; i++) {
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
    }
}



int main() {
    int ns = 10;

    int sizes[][2] = {
        {32, 32},
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512}
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    int thread_counts[] = {1, 2, 3, 4, 8, 16};
    int num_threads_options = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int s = 0; s < num_sizes; s++) {
        int w = sizes[s][0];
        int h = sizes[s][1];

        for (int t = 0; t < num_threads_options; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);

            int img_size = sizeof(unsigned char) * w * h * 3;
            unsigned char* data = (unsigned char*)calloc(img_size, 1);

            std::clock_t start = std::clock();

            rayTracingCPU(data, w, h, ns, 0, 0, w, h); // OpenMP actúa dentro

            std::clock_t end = std::clock();
            double duration = double(end - start) / CLOCKS_PER_SEC;

            printf("Tiempo total para %dx%d con %d hilos: %.3f segundos\n",
                   w, h, num_threads, duration);

            char filename[128];
            sprintf(filename, "openmp_%dx%d_threads%d.bmp", w, h, num_threads);
            writeBMP(filename, data, w, h);

            free(data);
        }
    }

    return 0;
}