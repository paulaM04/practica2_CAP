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

Scene randomScene()
{
    int n = 500;
    Scene list;
    list.add(new Object(
        new Sphere(Vec3(0, -1000, 0), 1000),
        new Diffuse(Vec3(0.5, 0.5, 0.5))));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = RayTracingCPU::random();
            Vec3 center(a + 0.9f * RayTracingCPU::random(), 0.2f, b + 0.9f * RayTracingCPU::random());
            if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f)
            {
                if (choose_mat < 0.8f)
                { // diffuse
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Diffuse(Vec3(RayTracingCPU::random() * RayTracingCPU::random(),
                                         RayTracingCPU::random() * RayTracingCPU::random(),
                                         RayTracingCPU::random() * RayTracingCPU::random()))));
                }
                else if (choose_mat < 0.95f)
                { // metal
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Metallic(Vec3(0.5f * (1 + RayTracingCPU::random()),
                                          0.5f * (1 + RayTracingCPU::random()),
                                          0.5f * (1 + RayTracingCPU::random())),
                                     0.5f * RayTracingCPU::random())));
                }
                else
                { // glass
                    list.add(new Object(
                        new Sphere(center, 0.2f),
                        new Crystalline(1.5f)));
                }
            }
        }
    }

    list.add(new Object(
        new Sphere(Vec3(0, 1, 0), 1.0),
        new Crystalline(1.5f)));
    list.add(new Object(
        new Sphere(Vec3(-4, 1, 0), 1.0f),
        new Diffuse(Vec3(0.4f, 0.2f, 0.1f))));
    list.add(new Object(
        new Sphere(Vec3(4, 1, 0), 1.0f),
        new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)));

    return list;
}



void rayTracingCPU(unsigned char *img, int w, int h, int ns, int px, int py, int pw, int ph)
{
    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;
    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    for (int j = 0; j < ph - py; j++)
    {
        for (int i = 0; i < pw - px; i++)
        {
            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++)
            {
                float u = float(i + px + RayTracingCPU::random()) / float(w);
                float v = float(j + py + RayTracingCPU::random()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            img[(j * pw + i) * 3 + 0] = static_cast<unsigned char>(255.99f * col[2]); // Azul
            img[(j * pw + i) * 3 + 1] = static_cast<unsigned char>(255.99f * col[1]); // Verde
            img[(j * pw + i) * 3 + 2] = static_cast<unsigned char>(255.99f * col[0]); // Rojo
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3)
    {
        if (rank == 0)
            printf("Uso: %s <ancho> <alto>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int w = atoi(argv[1]); 
    int h = atoi(argv[2]); 
    int ns = 10;

    int base_rows = h / size;
    int extra = h % size;

    int start_row = rank * base_rows + std::min(rank, extra);
    int num_rows = base_rows + (rank < extra ? 1 : 0);

    int local_pixels = w * num_rows * 3;
    unsigned char *local_img = (unsigned char *)calloc(local_pixels, 1);

    double start_time = MPI_Wtime();
    rayTracingCPU(local_img, w, h, ns, 0, start_row, w, start_row + num_rows);
    double end_time = MPI_Wtime();

    unsigned char *global_img = nullptr;
    int *recvcounts = nullptr;
    int *displs = nullptr;

    if (rank == 0)
    {
        global_img = (unsigned char *)malloc(w * h * 3);
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++)
        {
            int rows_i = base_rows + (i < extra ? 1 : 0);
            recvcounts[i] = rows_i * w * 3;
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    MPI_Gatherv(local_img, local_pixels, MPI_UNSIGNED_CHAR,
                global_img, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        char filename[128];
        sprintf(filename, "img_final_filas_%dx%d_MPI.bmp", w, h);
        writeBMP(filename, global_img, w, h);
        printf("Tiempo de CPU para %dx%d con %d procesos: %.3f segundos.\n", w, h, size, end_time - start_time);

        free(global_img);
        free(recvcounts);
        free(displs);
    }

    free(local_img);
    MPI_Finalize();
    return 0;
}
