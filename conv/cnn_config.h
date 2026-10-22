#ifndef CNN_CONFIG_H
#define CNN_CONFIG_H

/* CCS 3.1 / C6000 COFF, -mv6400, little endian. No BIOS required. */
#if defined(_TMS320C6X) || defined(__TMS320C6X__)
#define CNN_TI_TARGET 1
#else
#define CNN_TI_TARGET 0
#endif

/* The only C / assembly selection. Change this to 0 or 1, then Rebuild All.
 * Native host tests explicitly compile with -DUSE_LINEAR_ASM=0. */
#ifndef USE_LINEAR_ASM
#define USE_LINEAR_ASM 1
#endif
#if USE_LINEAR_ASM != 0 && USE_LINEAR_ASM != 1
#error USE_LINEAR_ASM must be 0 or 1
#endif
#if USE_LINEAR_ASM && !CNN_TI_TARGET
#error The real C64x .sa kernel requires the TI C6000 compiler
#endif

#ifndef CNN_USE_STDIO
#define CNN_USE_STDIO 1
#endif
#define CNN_DOT_LENGTH 128
#define CNN_DOT_VALID 125
#define CNN_WEIGHT_ROW_SHORTS 144
#define CNN_WEIGHT_OFFSET_SHORTS 8

#endif
