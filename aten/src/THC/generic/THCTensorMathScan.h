#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathScan.h"
#else

#if !defined (THC_REAL_IS_BOOL)

THC_API void THCTensor_(cumsum)(THCState *state, THCTensor *self, THCTensor *src, int dim);
THC_API void THCTensor_(cumprod)(THCState *state, THCTensor *self, THCTensor *src, int dim);

#endif

#endif
