# Algorithm `ChangeLog.md`
PG1.5
- **Hardware**: Genkai Node Group A (1 node, 1 core)
- **Module**: GCC 13.3.1

## Change Log

- Format: `ChangeLog_format.md`
- PM Override: `ChangeLog_format_PM_override.md`

### v1.4.0
**Change**: "Prefetch optimization for A/B tile loads"
**Result**: 862.2 GFLOPS (44% efficiency) - REGRESSION
**Comment**: "Prefetch interfered with HW prefetcher. Performance dropped from v1.3.0."

<details>

- **Generated**: `2026-01-01T19:00:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `gcc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16`
- [x] **job**
    - id: `4610372`
    - resource_group: `a-batch-low`
    - start_time: `completed`
    - end_time: `completed`
    - runtime_sec: `~30`
    - status: `completed`
- [x] **test**
    - status: `regression`
    - performance: `862.2 GFLOPS (4096x4096)`
    - unit: `GFLOPS`
    - efficiency: `44%`
    - note: `Worse than v1.3.0 (48%)`
- **params**:
    - BLOCK_K: `1536`
    - BLOCK_N: `480`
    - prefetch: `_MM_HINT_T0`
    - method: `Tiling_B + Prefetch`

</details>

### v1.3.0
**Change**: "AMX with proper syscall initialization"
**Result**: 928.0 GFLOPS (48% efficiency)
**Comment**: "Added arch_prctl syscall for AMX permission. Major breakthrough from 65 to 928 GFLOPS."

<details>

- **Generated**: `2026-01-01T18:50:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `gcc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16`
- [x] **job**
    - id: `4610361`
    - resource_group: `a-batch-low`
    - start_time: `2026-01-01T18:51:54`
    - end_time: `completed`
    - runtime_sec: `~60`
    - status: `completed`
- [x] **test**
    - status: `success`
    - performance: `928.0 GFLOPS (4096x4096)`
    - unit: `GFLOPS`
    - efficiency: `48%`
- **params**:
    - BLOCK_K: `1536`
    - BLOCK_N: `480`
    - method: `Tiling_B`
    - fix: `syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)`

</details>

### v1.2.0
**Change**: "AVX-512 FMA with register blocking"
**Result**: 65.0 GFLOPS (3.4% efficiency)
**Comment**: "6x32 microkernel using AVX-512 FMA. Better than v1.1.0 but far from AMX potential."

<details>

- **Generated**: `2026-01-01T18:30:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `gcc -O3 -march=sapphirerapids`
- [x] **job**
    - id: `4610355`
    - resource_group: `a-batch-low`
    - status: `completed`
- [x] **test**
    - status: `success`
    - performance: `65.0 GFLOPS`
    - unit: `GFLOPS`
    - efficiency: `3.4%`
- **params**:
    - BLOCK_M: `48`
    - BLOCK_N: `256`
    - BLOCK_K: `512`
    - MR: `6`
    - NR: `32`
    - method: `AVX-512 FMA`

</details>

### v1.1.0
**Change**: "AVX-512 BF16 implementation"
**Result**: 30.0 GFLOPS (1.6% efficiency)
**Comment**: "Initial AVX-512 attempt with BF16 conversion. Low efficiency due to suboptimal packing."

<details>

- **Generated**: `2026-01-01T18:15:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
- [x] **job**
    - id: `4610352`
    - resource_group: `a-batch-low`
    - status: `completed`
- [x] **test**
    - status: `success`
    - performance: `30.0 GFLOPS`
    - unit: `GFLOPS`
    - efficiency: `1.6%`
- **params**:
    - method: `AVX-512 BF16`

</details>

### v1.0.0
**Change**: "Intel AMX Tiling_B initial implementation"
**Result**: FAILED (Illegal instruction)
**Comment**: "AMX requires kernel permission via arch_prctl syscall. Fixed in v1.3.0."

<details>

- **Generated**: `2026-01-01T09:43:26Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `gcc -O3 -march=sapphirerapids -mamx-tile -mamx-bf16`
- [x] **job**
    - id: `4610348`
    - resource_group: `a-batch-low`
    - status: `failed`
- [x] **test**
    - status: `failed`
    - error: `Illegal instruction (core dumped)`
    - cause: `Missing AMX permission syscall`
- **params**:
    - BLOCK_K: `1536`
    - BLOCK_N: `480`
    - method: `Tiling_B`

</details>
