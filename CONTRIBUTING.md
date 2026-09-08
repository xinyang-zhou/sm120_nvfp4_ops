# Contributing

Contributions should keep the three operator layers independently testable.

Before submitting a change:

1. build with `./scripts/build.sh`;
2. run `./scripts/test.sh` on an SM120 GPU;
3. add a correctness test for new shapes or layouts;
4. include benchmark methodology for performance claims;
5. avoid checking generated binaries, build directories or local dependency paths into the repository.

Kernel changes should document tile shape, pipeline stages, shared-memory use and the workload they target. Performance changes must report both latency and throughput and must not regress correctness or unsupported-shape handling.

Public API changes should preserve stream ordering and avoid hidden synchronization. New temporary storage should use an explicit workspace contract or a caller-owned cache.
