#include "inutils.h"

void set_CNN(int col, int cnn_num, char *tmp, cnn_t *cnn) {
  switch(col) {
    case 0:
      cnn[cnn_num].layer  = atoi(tmp);
      break;
    case 1: //M
      cnn[cnn_num].mmin  = atoi(tmp);
      cnn[cnn_num].mmax  = atoi(tmp);
      cnn[cnn_num].mstep = 1;
      break;
    case 2: //N
      cnn[cnn_num].nmin  = atoi(tmp);
      cnn[cnn_num].nmax  = atoi(tmp);
      cnn[cnn_num].nstep = 1;
      break;
  case 3: //K
      cnn[cnn_num].kmin  = atoi(tmp);
      cnn[cnn_num].kmax  = atoi(tmp);
      cnn[cnn_num].kstep = 1;
      break;
  }
}


testConfig_t* new_CNN_Test_Config(char * argv[]) {
  FILE *fd_conf = fopen(argv[21], "r"); //open config file
  char * line = NULL;
  size_t len = 0;
  size_t read;
  const char delimiter[] = "\t";
  char *tmp;
  int col;
  testConfig_t *new_testConfig = (testConfig_t *)malloc(sizeof(testConfig_t));
  int cnn_num;
  
  new_testConfig->tmin   = 0;
  new_testConfig->test   = 0;
  new_testConfig->debug  = 0;
  
  cnn_num=0;    
  while ((read = getline(&line, &len, fd_conf)) != -1)
    if (line[0] != '#') {      
      col = 0;
      tmp = strtok(line, delimiter);
      if (tmp == NULL)
	break;
      set_CNN(col, cnn_num, tmp, new_testConfig->cnn);
      col++;
      for (;;) {
	tmp = strtok(NULL, delimiter);
	if (tmp == NULL)
	  break;
	set_CNN(col, cnn_num, tmp, new_testConfig->cnn);
	col++;
      }

      cnn_num++;
    }

  fclose(fd_conf); 

  new_testConfig->cnn_num = cnn_num;
  
  return new_testConfig;
}

void free_CNN_Test_Config(testConfig_t *testConfig) {
  free(testConfig);
}
