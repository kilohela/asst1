
export void saxpy_ispc(uniform int N,
                       uniform float scale,
                            uniform float X[],
                            uniform float Y[],
                            uniform float result[])
{
    foreach (i = 0 ... N) {           
        result[i] = scale * X[i] + Y[i];
    }
}

task void saxpy_ispc_task(uniform int N,
                               uniform int span,
                               uniform float scale,
                               uniform float X[], 
                               uniform float Y[],
                               uniform float result[])
{

    uniform int indexStart = taskIndex * span;
    uniform int indexEnd = min(N, indexStart + span);

    // 每次处理一个“gang”（programCount 个元素）
    for (uniform int base = indexStart; base < indexEnd; base += programCount) {
        int i = base + programIndex;

        // 越界 lane 关掉（最后一个 gang 可能不满）
        if (i < indexEnd) {
            float r = scale * X[i] + Y[i];
            // non-temporal / streaming store
            streaming_store(&result[base], r);
        }
    }
}

export void saxpy_ispc_withtasks(uniform int N,
                               uniform float scale,
                               uniform float X[],
                               uniform float Y[],
                               uniform float result[])
{
    uniform int span = ((N / 64 + 31) / 32) * 32;  // round up to multiple of 32
    if (span < 32) span = 32;

    launch[(N+span-1)/span] saxpy_ispc_task(N, span, scale, X, Y, result);
}
