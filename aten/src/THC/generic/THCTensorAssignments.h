#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorAssignments.h"
#else

THC_API void THCTensor_(fill)(THCState *state, THCTensor *self, scalar_t value);
THC_API void THCTensor_(zero)(THCState *state, THCTensor *self);

THC_API void THCTensor_(eye)(THCState *state, THCTensor *self, int64_t n, int64_t k);

#endif
