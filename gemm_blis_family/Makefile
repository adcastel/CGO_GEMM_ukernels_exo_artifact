DTYPE = -DFP32
MR = 8
NR = 12

BLIS_HOME    = ${BLISHOME}
INCLUDE_BLIS = -I$(BLIS_HOME)/include/blis/
LIB_BLIS     = -lblis -L$(BLIS_HOME)/lib

INCLUDE = ${INCLUDE_BLIS}
LIBS    = ${LIB_BLIS} -lm


#------------------------------------------
#| COMPILERS                              |
#------------------------------------------
CC       =  gcc
CLINKER  =  gcc
#------------------------------------------

OBJDIR = build
BIN    = test_gemm.x
_OBJ   = gemm_blis.o inutils.o sutils.o test_gemm.o model_level.o 

#------------------------------------------
#| PATHS CONFIGURE                        |
#------------------------------------------
vpath %.c ./src
vpath %.h ./src

vpath %.c ./modelLevel
vpath %.h ./modelLevel

vpath %.c ./src/ARMv8
vpath %.h ./src/ARMv8
#------------------------------------------

#------------------------------------------
#| COMPILER FLAGS                         |
#------------------------------------------

ifeq ($(RUN_MODE), FAMILY)
  MODE=-DFAMILY
else ifeq ($(RUN_MODE), FAMILY_BLIS)
  MODE=-DFAMILY_BLIS
else ifeq ($(RUN_MODE), FAMILY_EXO)
  MODE=-DFAMILY_EXO
else ifeq ($(RUN_MODE), BLIS)
  MODE=-DBLIS
endif

#Dependes Arquitecture Mode
ifeq ($(SIMD_MODE), ARMv8)
  SIMD  = -DARMv8
  FLAGS = -O3 -march=armv8-a+simd+fp -fopenmp
  _OBJ += gemm_blis_neon_fp32.o 
else ifeq ($(SIMD_MODE), ARMv8_EXO)
  SIMD  = -DARMv8_EXO
  FLAGS = -O3 -march=armv8-a+simd+fp
  _OBJ += uk_exo.o 
else
  FLAGS = -O3 
endif

ifeq ($(RUN_MODE), FAMILY_BLIS)
  # FIX for all BLIS MR,NR Values!
  ifeq ($(SIMD_MODE), ARMv8)
    MR=8
    NR=12
  else ifeq ($(SIMD_MODE), ARMv8_EXO)
    MR=8
    NR=12
  else
  endif	
endif

OPTFLAGS = $(FLAGS) -DCHECK -DMR=$(MR) -DNR=$(NR) -DKR=1 $(MODE) $(SIMD) $(DTYPE)
#------------------------------------------

OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))
default: $(OBJDIR)/$(BIN)

$(OBJDIR)/%.o:%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c -o $@ $< $(INCLUDE) $(LIBS)

$(OBJDIR)/$(BIN): $(OBJ) 
	$(CLINKER) $(OPTFLAGS) -o $@ $^ $(LIBS)

clean:
	rm $(OBJDIR)/* 

