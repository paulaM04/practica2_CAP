#include <float.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include "Camera.hpp"
#include "Object.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"
#include "Diffuse.hpp"
#include "Metallic.hpp"
#include "Crystalline.hpp"
#include "Random.hpp"
#include "Utils.hpp"

Scene randomScene()
{
    Scene list;
    list.add(new Object(new Sphere(Vec3(0, -1000, 0), 1000), new Diffuse(Vec3(0.5, 0.5, 0.5))));

    for (int a = -11; a < 11; a++)
    {
        for (int b = -11; b < 11; b++)
        {
            float choose_mat = RayTracingCPU::random();
            Vec3 center(a + 0.9f * RayTracingCPU::random(), 0.2f, b + 0.9f * RayTracingCPU::random());
            if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f)
            {
                if (choose_mat < 0.8f)
                {
                    list.add(new Object(new Sphere(center, 0.2f),
                                        new Diffuse(Vec3(RayTracingCPU::random() * RayTracingCPU::random(),
                                                         RayTracingCPU::random() * RayTracingCPU::random(),
                                                         RayTracingCPU::random() * RayTracingCPU::random()))));
                }
                else if (choose_mat < 0.95f)
                {
                    list.add(new Object(new Sphere(center, 0.2f),
                                        new Metallic(Vec3(0.5f * (1 + RayTracingCPU::random()),
                                                          0.5f * (1 + RayTracingCPU::random()),
                                                          0.5f * (1 + RayTracingCPU::random())),
                                                     0.5f * RayTracingCPU::random())));
                }
                else
                {
                    list.add(new Object(new Sphere(center, 0.2f),
                                        new Crystalline(1.5f)));
                }
            }
        }
    }

    list.add(new Object(new Sphere(Vec3(0, 1, 0), 1.0), new Crystalline(1.5f)));
    list.add(new Object(new Sphere(Vec3(-4, 1, 0), 1.0f), new Diffuse(Vec3(0.4f, 0.2f, 0.1f))));
    list.add(new Object(new Sphere(Vec3(4, 1, 0), 1.0f), new Metallic(Vec3(0.7f, 0.6f, 0.5f), 0.0f)));

    return list;
}

void rayTracingCPU(unsigned char *img, int w, int h, int ns, int px, int pw)
{
    int local_cols = pw - px;

    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;
    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    for (int j = 0; j < h; j++)
    {
        for (int i = 0; i < local_cols; i++)
        {
            int global_i = px + i;
            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++)
            {
                float u = float(global_i + RayTracingCPU::random()) / float(w);
                float v = float(j + RayTracingCPU::random()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            int index = (j * local_cols + i) * 3;
            img[index + 0] = static_cast<unsigned char>(255.99f * col[2]);
            img[index + 1] = static_cast<unsigned char>(255.99f * col[1]);
            img[index + 2] = static_cast<unsigned char>(255.99f * col[0]);
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

    int base_cols = w / size;
    int extra = w % size;

    int start_col = rank * base_cols + std::min(rank, extra);
    int num_cols = base_cols + (rank < extra ? 1 : 0);

    int local_pixels = num_cols * h * 3;
    unsigned char *local_img = (unsigned char *)calloc(local_pixels, 1);
    if (!local_img)
    {
        fprintf(stderr, "Proceso %d: error al reservar memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double start_time = MPI_Wtime();
    rayTracingCPU(local_img, w, h, ns, start_col, start_col + num_cols);
    double end_time = MPI_Wtime();


    unsigned char *global_img = nullptr;
    if (rank == 0)
        global_img = (unsigned char *)malloc(w * h * 3);

    int *recv_counts = (int *)malloc(size * sizeof(int));
    int *col_starts = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i)
    {
        int cols = base_cols + (i < extra ? 1 : 0);
        recv_counts[i] = cols * h * 3;
        col_starts[i] = i * base_cols + std::min(i, extra);
    }

    unsigned char *all_data = nullptr;
    if (rank == 0)
        all_data = (unsigned char *)malloc(w * h * 3);

    MPI_Gather(local_img, local_pixels, MPI_UNSIGNED_CHAR,
               all_data, local_pixels, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
   
        for (int i = 0, offset = 0; i < size; i++)
        {
            int cols = base_cols + (i < extra ? 1 : 0);
            int col_start = col_starts[i];
            for (int row = 0; row < h; row++)
            {
                for (int col = 0; col < cols; col++)
                {
                    int src_idx = (row * cols + col) * 3;
                    int dst_idx = (row * w + (col_start + col)) * 3;
                    global_img[dst_idx + 0] = all_data[offset + src_idx + 0];
                    global_img[dst_idx + 1] = all_data[offset + src_idx + 1];
                    global_img[dst_idx + 2] = all_data[offset + src_idx + 2];
                }
            }
            offset += cols * h * 3;
        }

        char filename[128];
        sprintf(filename, "img_final_columnas_%dx%d_MPI.bmp", w, h);
        writeBMP(filename, global_img, w, h);
        printf("Tiempo de CPU para %dx%d con %d procesos: %.3f segundos.\n",
               w, h, size, end_time - start_time);

        free(global_img);
        free(all_data);
    }

    free(local_img);
    free(recv_counts);
    free(col_starts);

    MPI_Finalize();
    return 0;
}
