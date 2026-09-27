#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <vector>

int main() {
  constexpr int G=2, M=16, N=4096, K=8192;
  cublasLtHandle_t h{}; cublasLtMatmulDesc_t op{}; cublasLtMatrixLayout_t a{},b{},d{};
  auto ck=[&](cublasStatus_t s,const char*n){if(s!=CUBLAS_STATUS_SUCCESS){printf("%s status=%d\n",n,(int)s); return false;}return true;};
  if(!ck(cublasLtCreate(&h),"create")) return 1;
  if(!ck(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"desc")) return 1;
  cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta));
  cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb));
  int *ra,*ca,*la,*rb,*cb,*lb,*rd,*cd,*ld; cudaMalloc(&ra,G*4);cudaMalloc(&ca,G*4);cudaMalloc(&la,G*4);cudaMalloc(&rb,G*4);cudaMalloc(&cb,G*4);cudaMalloc(&lb,G*4);cudaMalloc(&rd,G*4);cudaMalloc(&cd,G*4);cudaMalloc(&ld,G*4);
  std::vector<int> vra(G,K),vca(G,M),vla(G,K),vrb(G,K),vcb(G,N),vlb(G,K),vrd(G,M),vcd(G,N),vld(G,M);
  cudaMemcpy(ra,vra.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(ca,vca.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(la,vla.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(rb,vrb.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(cb,vcb.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(lb,vlb.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(rd,vrd.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(cd,vcd.data(),G*4,cudaMemcpyHostToDevice);cudaMemcpy(ld,vld.data(),G*4,cudaMemcpyHostToDevice);
  bool ok=true; ok &= ck(cublasLtGroupedMatrixLayoutCreate(&a,CUDA_R_4F_E2M1,G,ra,ca,la),"layoutA"); ok &= ck(cublasLtGroupedMatrixLayoutCreate(&b,CUDA_R_4F_E2M1,G,rb,cb,lb),"layoutB"); ok &= ck(cublasLtGroupedMatrixLayoutCreate(&d,CUDA_R_16F,G,rd,cd,ld),"layoutD");
  if(ok) { int count=0; cublasLtMatmulPreference_t p{}; cublasLtMatmulPreferenceCreate(&p); size_t ws=64<<20; cublasLtMatmulPreferenceSetAttribute(p,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)); cublasLtMatmulHeuristicResult_t r[32]{}; auto s=cublasLtMatmulAlgoGetHeuristic(h,op,a,b,d,d,p,32,r,&count); printf("heuristic status=%d count=%d\n",(int)s,count); if(count){int cap=0; size_t written=0; cublasLtMatmulAlgoCapGetAttribute(&r[0].algo,CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT,&cap,sizeof(cap),&written); printf("grouped_cap=%d\n",cap);}}
  cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(d);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h); return 0;
}
