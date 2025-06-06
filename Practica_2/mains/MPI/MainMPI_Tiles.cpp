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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mpi.h>


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

void rayTracingCPU(unsigned char *img, int w, int h, int ns, int px, int py, int pw, int ph)
{
    int tile_w = pw - px;
    int tile_h = ph - py;

    Scene world = randomScene();
    world.setSkyColor(Vec3(0.5f, 0.7f, 1.0f));
    world.setInfColor(Vec3(1.0f, 1.0f, 1.0f));

    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0;
    float aperture = 0.1f;

    Camera cam(lookfrom, lookat, Vec3(0, 1, 0), 20, float(w) / float(h), aperture, dist_to_focus);

    for (int j = 0; j < tile_h; j++)
    {
        for (int i = 0; i < tile_w; i++)
        {
            int global_i = px + i;
            int global_j = py + j;
            Vec3 col(0, 0, 0);
            for (int s = 0; s < ns; s++)
            {
                float u = float(global_i + RayTracingCPU::random()) / float(w);
                float v = float(global_j + RayTracingCPU::random()) / float(h);
                Ray r = cam.get_ray(u, v);
                col += world.getSceneColor(r);
            }
            col /= float(ns);
            col = Vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

            int index = (j * tile_w + i) * 3;
            img[index + 0] = static_cast<unsigned char>(255.99f * col[2]); // azul
            img[index + 1] = static_cast<unsigned char>(255.99f * col[1]); // verde
            img[index + 2] = static_cast<unsigned char>(255.99f * col[0]); // rojo
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

    int tiles_x = (int)sqrt(size);
    while (size % tiles_x != 0)
        tiles_x--;
    int tiles_y = size / tiles_x;

    int tile_x = rank % tiles_x;
    int tile_y = rank / tiles_x;

    int tile_w = w / tiles_x;
    int tile_h = h / tiles_y;
    int extra_w = w % tiles_x;
    int extra_h = h % tiles_y;

    int px = tile_x * tile_w + std::min(tile_x, extra_w);
    int pw = px + tile_w + (tile_x < extra_w ? 1 : 0);
    int py = tile_y * tile_h + std::min(tile_y, extra_h);
    int ph = py + tile_h + (tile_y < extra_h ? 1 : 0);

    int local_w = pw - px;
    int local_h = ph - py;
    int local_pixels = local_w * local_h * 3;

    unsigned char *local_img = (unsigned char *)calloc(local_pixels, 1);
    if (!local_img)
    {
        fprintf(stderr, "Proceso %d: error al reservar memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t0 = MPI_Wtime();
    rayTracingCPU(local_img, w, h, ns, px, py, pw, ph);
    double t1 = MPI_Wtime();

    unsigned char *global_img = nullptr;
    if (rank == 0)
        global_img = (unsigned char *)calloc(w * h * 3, 1);


    if (rank == 0)
    {

        for (int j = 0; j < local_h; j++)
        {
            for (int i = 0; i < local_w; i++)
            {
                int src = (j * local_w + i) * 3;
                int dst = ((py + j) * w + (px + i)) * 3;
                global_img[dst + 0] = local_img[src + 0];
                global_img[dst + 1] = local_img[src + 1];
                global_img[dst + 2] = local_img[src + 2];
            }
        }

        for (int p = 1; p < size; p++)
        {
            int tile_xp = p % tiles_x;
            int tile_yp = p / tiles_x;
            int pxp = tile_xp * tile_w + std::min(tile_xp, extra_w);
            int pwp = pxp + tile_w + (tile_xp < extra_w ? 1 : 0);
            int pyp = tile_yp * tile_h + std::min(tile_yp, extra_h);
            int php = pyp + tile_h + (tile_yp < extra_h ? 1 : 0);

            int lw = pwp - pxp;
            int lh = php - pyp;
            int npix = lw * lh * 3;
            unsigned char *recv_buf = (unsigned char *)malloc(npix);
            MPI_Recv(recv_buf, npix, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 0; j < lh; j++)
            {
                for (int i = 0; i < lw; i++)
                {
                    int src = (j * lw + i) * 3;
                    int dst = ((pyp + j) * w + (pxp + i)) * 3;
                    global_img[dst + 0] = recv_buf[src + 0];
                    global_img[dst + 1] = recv_buf[src + 1];
                    global_img[dst + 2] = recv_buf[src + 2];
                }
            }

            free(recv_buf);
        }

        char filename[128];
        sprintf(filename, "img_final_tiles_%dx%d_MPI.bmp", w, h);
        writeBMP(filename, global_img, w, h);
        printf("Tiempo de CPU para %dx%d con %d procesos: %.3f segundos.\n", w, h, size, t1 - t0);

    }
    else
    {
        MPI_Send(local_img, local_pixels, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    free(local_img);
    MPI_Finalize();
    return 0;
}
