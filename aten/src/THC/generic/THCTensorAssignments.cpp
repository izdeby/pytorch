#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorAssignments.cpp"
#else

void THCTensor_(fill)(THCState* state, THCTensor *self_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));

  if (!THC_pointwiseApply1<scalar_t>(
        state, self_, TensorFillOp<scalar_t>(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(zero)(THCState *state, THCTensor *self_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  if (THCTensor_(isContiguous)(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCTensor_(data)(state, self_),
                                0,
                                sizeof(scalar_t) * THCTensor_(nElement)(state, self_),
                                THCState_getCurrentStream(state)));
  } else {
    if (!THC_pointwiseApply1<scalar_t>(
          state, self_,
          TensorFillOp<scalar_t>(ScalarConvert<int, scalar_t>::to(0)))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(eye)(THCState *state, THCTensor *self_, int64_t n, int64_t m)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THCTensor_(resize2d)(state, self_, n, m);
  THCTensor_(zero)(state, self_);

  int64_t sz = THMin(n, m);
  int64_t stride = THCTensor_(stride)(state, self_, 0) +
                   THCTensor_(stride)(state, self_, 1);

  THCTensor *diag = THCTensor_(newWithStorage1d)(state, THTensor_getStoragePtr(self_),
      self_->storage_offset(),  sz, stride);

  THCTensor_(fill)(state, diag, ScalarConvert<int, scalar_t>::to(1));
  THCTensor_(free)(state, diag);
}
#endif
