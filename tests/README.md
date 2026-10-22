# 本机独立验证

需要 Python 3 与 clang；不需要 TensorFlow、NumPy 或 DSP 板。

在项目根目录执行：

```sh
python3 tests/verify_host.py
python3 tests/verify_sa_semantics.py
python3 tests/verify_call_route.py
```

`verify_host.py` 在临时目录生成 500 张图像的数据头，分别编译 `USE_LINEAR_ASM=0` 和 `=1`，并启用 AddressSanitizer、UndefinedBehaviorSanitizer。第二种构建链接明确标注的宿主机 C 桩，检查每张图实际经过外部 `cnn_dotprod128` 符号 640 次。两个构建均与独立编译的原始 `reference/main.c` 逐项比较所有中间层、最终分数和预测，并检查：

- 500 张 MNIST 图像，以及 67 个全零、全一、交错和随机二值输入。
- 128 项 C 点积，打包权重的通道顺序、三个零补位和 `x%32=0`、`w%32=16` 合同。
- 原前 20 张预测，以及 500 张准确率 485/500。
- 非法输入和空指针被拒绝；输出状态事先污染后仍被完整写入。
- 交付权重与图像头匹配原始文本，golden 的 COUNT/OFFSET 和完整数据头 SHA256 一致。

原始 C 对照仅在临时副本中重命名符号、替换数据路径，不改卷积和池化数学；每张图的 argmax 从第 0 类开始。宿主机 `long` 比 TI COFF 的 40 位 `long` 宽，新版推理另有数值范围检查。macOS 不支持 LeakSanitizer，因此只在 macOS 关闭泄漏检测；地址和未定义行为检查仍开启。

`verify_sa_semantics.py` 读取实际 `conv/cnn_dotprod128.sa`，解释寄存器对、`LDDW`、有符号 `DOTP2`、双累加器、循环及返回的数据语义。它跟踪两个缓冲区的每个字节，要求 128 个 short 均恰好读取一次，并检查单项、部分和与最终和没有 signed32 溢出。1164 个用例覆盖负数、16 位边界、真实模型权重、三个零补位，以及 128 个逐位置单独激活的输入。每个结果同时与 Python 精确整数及编译的 C 点积比较；遇到未知指令立即失败。

`verify_call_route.py` 编译真实的 `main.c`、`clock_demo.c` 和两种内核选择。每种构建都在污染全局报告、状态和 Clock 变量后连续重入主程序两次，检查 19/20、全部 200 个 golden 分数、显式清零、首图 demo patch 保存和对齐合同。汇编分支使用宿主机 C 桩，核对主程序的 12800 次卷积调用、一次区外预热和一次 Clock 演示调用。脚本同时静态检查 VECS 保留 `[0,0x400)`、IRAM 从 `0x400` 开始。

`generate_golden.py` 独立编译原始数学代码，生成 `conv/cnn_golden.h` 的全部分数与预测，不链接生产 `cnn.c` 或 `.sa`。更换嵌入数据、权重或数据头后，先显式重新生成：

```sh
python3 tests/generate_golden.py
```

完整数据头 SHA256 包含权重、图像、标签和格式；头文件发生任何变化，PC 验证均要求重新生成 golden。主程序通过 COUNT/OFFSET 和逐图 10 个分数检查，不在 DSP 上运行 SHA256。主程序回归脚本专门固定检查交付的前 20 张图；若有意更换样本范围，相应预期也需要更新。

这些脚本**没有汇编或执行真正的 DSP 汇编**，也不验证 TI 汇编优化器、寄存器分配、COFF ABI、流水线时序或硬件周期。`-D_TMS320C6X=1` 只让宿主机编译测试调用分支。实际课程工具链构建、板卡运行和 CCS Clock 测量仍需单独完成。

验证脚本只在系统临时目录生成编译产物，不修改交付数据或源码；只有显式执行 `generate_golden.py` 才会更新 golden 头。
