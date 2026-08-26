# rocprofv3 collection failure

Two attempts were made with ROCm 7.2.2 `rocprofv3 1.1.0` on the gate-up
`[32,5760,2880]` quant workload. The second attempt also enabled
`--privileged`, `seccomp=unconfined`, and `SYS_PTRACE`.

Representative command:

```bash
rocprofv3 --kernel-trace --stats --output-format csv \
  --output-directory <out> --output-file quant_2d_direct -- \
  python profile_quant.py --mode 2d_direct --shape 32,5760,2880 \
  --warmup 5 --iterations 10
```

The target workload completed, but rocprofv3 aborted while finalizing and did
not emit a valid output package:

```text
ring_buffer: munmap failed: Invalid argument
ring_buffer.cpp:106] mmap failed with errno 22 :: Invalid argument
rocprofv3 caught signal 6
```

The profiling container was stopped after it remained stuck in its signal
handler. PyTorch ROCTracer was used as the fallback; its tables and Chrome
traces are stored in this directory.
