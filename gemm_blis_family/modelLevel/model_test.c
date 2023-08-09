#include <stdio.h>
#include "model_level.h"

#define RESNET50_LAYERS 20

#define MR 16
#define NR 14

int main(int argc, char *argv[]) {
  
  const int nLayers = RESNET50_LAYERS;
  int mc = 0, nc = 0, kc = 0;
  int m, n, k;

  int resNet50[RESNET50_LAYERS][4] = {
  {1,  1605632, 64,   147},
  {2,  401408,  64,   64},
  {3,  401408,  64,   576},
  {4,  401408,  256,  64},
  {5,  401408,  64,   256},
  {6,  401408,  128,  256},
  {7,  100352,  128,  1152},
  {8,  100352,  512,  128},
  {9,  100352,  512,  256},
  {10, 100352,  128,  512},
  {11, 100352,  256,  512},
  {12, 25088,   256,  2304},
  {13, 25088,   1024, 256},
  {14, 25088,   1024, 512},
  {15, 25088,   256,  1024},
  {16, 25088,   512,  1024},
  {17, 6272,    512,  4608},
  {18, 6272,    2048, 512},
  {19, 6272,    2048, 1024},
  {20, 6272,    512,  2048}
  };

  printf("\n");
  printf("-------------------------------------------------\n");
  printf("| MODEL LEVEL TEST CONFIGURATION                |\n");
  printf("-------------------------------------------------\n");
  printf("| [*] CNN MODEL: %-30s |\n", "ResNet50v15_imagenet");
  printf("| [*] ARCH     : %-30s |\n", ARCH_NAME);
  printf("| [*] MR       : %-30d |\n", MR);
  printf("| [*] NR       : %-30d |\n", NR);
  printf("-------------------------------------------------\n");
  printf("\n");

  printf("---------------------------------------------------------------\n");
  printf("| #l |    M          N        K   |      MC      NC        KC |\n");
  printf("---------------------------------------------------------------\n");
  for (int l = 0; l < nLayers; l++) {
    m = resNet50[l][1];
    n = resNet50[l][2];
    k = resNet50[l][3];

    get_optim_mc_nc_kc(sizeof(float), m, n, k, MR, NR, &mc, &nc, &kc);
    printf("|%3d |%8d %8d %8d  |%8d %8d %8d |\n", l, m, n, k, mc, nc, kc);
  }
  printf("---------------------------------------------------------------\n");

}
