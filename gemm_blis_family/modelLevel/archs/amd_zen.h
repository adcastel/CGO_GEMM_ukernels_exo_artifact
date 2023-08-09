#ifndef AMD_ZEN_H
#define AMD_ZEN_H

//---------------------------------------------------+
//| MODELS LEVELS CONFIGURATION FOR 'AMD ZEN'        |
//+--------------------------------------+-----------+
//| Note: CLn = SLn / (WLn * NLn)        |
//+--------------------------------------+
  #define ARCH_NAME "AMD ZEN"
//+--------------------------------------+
//| [*] Cache Level-1                    |
//+--------------------------------------+
  #define SL1 32768
  #define NL1 64
  #define CL1 64
  #define WL1 8
//+--------------------------------------+
//| [*] Cache Level-2                    |
//+--------------------------------------+
  #define SL2 524288
  #define NL2 1024
  #define CL2 64
  #define WL2 8
//+--------------------------------------+
//| [*] Cache Level-3                    |
//+--------------------------------------+
  #define SL3 16777216
  #define NL3 16384
  #define CL3 64
  #define WL3 16
//---------------------------------------------------+
//                                                   |
//---------------------------------------------------+

#endif
