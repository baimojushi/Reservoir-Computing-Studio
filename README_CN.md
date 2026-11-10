# RC Studio V0.4 Learning

新增悬停全文帮助、F1 学习中心与新手实操课程。先阅读 [本轮使用与验收说明](LEARNING_GUIDE_CN.md)。

**本轮验证边界：已完成纯数据/状态机/回归测试，原生 Qt 与 Windows 实机验收仍待完成。**

---

以下保留原版本说明与启动信息。

# RC Studio V0.3 — Material 3 风格储备池计算研究工作台

**Windows 笔记本本地运行 · PySide6 GUI · NumPy/SciPy ESN · PowerShell CLI · 127.0.0.1 HTTP · MCP v2 stdio**

本版直接在上传的 **Reservoir-Computing-Studio-main V0.1 源码**上修复与增补；保留通用单变量 ESN 项目，新增 **EBrush Motion Predictor V1** 离线六通道研究项目，并形成 E-Brush Studio 与 RC Studio 的清晰职责边界。当前交付内容为可独立运行的完整源码工程。

> **验收状态**：NumPy/SciPy 计算、E-Brush BRLOG 解码、导入、分记录隔离、服务鉴权、CLI 训练与测试已在容器执行；**本环境未安装 PySide6、Windows、真实毛笔硬件及 MCP 客户端，GUI 观感、Windows 启动和 MCP 第三方客户端连接仍需目标机验收**。电子毛笔模型目前作为研究基线使用；商用笔尖轨迹精度需要后续实机验证。

## 1. 职责边界（不可跨越）

| 模块 | E-Brush Studio（8765） | RC Studio（8766） |
|---|---|---|
| USB/COM 串口、BLE、设备命令 | 唯一负责 | **禁止连接** |
| 传感器采集、相机与姿态标定 | 负责 | 不实现 |
| 原始 `.brlog`/`.csv` 归档 | 负责 | **只读本地文件，派生自己的训练缓存** |
| 解析离线日志、分割样本、数据质量检查 | 可提供实时采集质量快照 | **负责历史文件离线分析** |
| ESN 训练、持续性基线、模型、实验 DB | 不实现 | **独立负责** |
| 外部 Agent | 设备/标定能力 | 只读实验和训练工具；写工具单独显式授权 |

两个服务独立使用不同端口和工作目录。RC Studio **没有**任何串口、摄像头或直接调用 E-Brush `/POST` 的代码。

## 2. Windows 安装与启动

推荐 Windows 10/11 64 位、Python 3.11+，Windows PowerShell。从解压后工程根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

也可用不弹控制台的窗口启动器：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\launch_gui.ps1
```

`launch_gui.ps1` 不再强制结束现有 Python / RC Studio / E-Brush 进程，也不写死开发者的 D: 盘路径。`scripts/start.ps1` 使用 RC Studio 自己的 `.venv`、自动连接独立本机服务，默认端口 **8766**。

不安装 GUI 依赖时可只用命令行：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m rcstudio check
```

### 无硬件验证 EBrush 内置项目

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ebrush.ps1 -Demo -Train
```

或：

```powershell
$RC = ".\.venv\Scripts\python.exe"
& $RC -m rcstudio project ebrush demo --train
& $RC -m rcstudio project ebrush sessions
& $RC -m rcstudio project ebrush runs
```

演示命令会新建 **6 份独立的合成录制**并对该批会话进行六通道训练、验证和测试，明确在指标中标记 `synthetic_included=true`。这仅验证软件流程，不能用作真实电子毛笔精度证据。

### 导入真实电子毛笔数据

支持 `.brlog`、6 列物理量/原始码的 `.csv`、含 `.brlog` 的项目 `.zip`、E-Brush 生成的 `ebrush.handoff.v1` `.json`：

```powershell
& $RC -m rcstudio project ebrush import "D:\EBrush\logs\session_A.brlog"
& $RC -m rcstudio project ebrush import "D:\EBrush\logs\session_B.brlog"
& $RC -m rcstudio project ebrush import "D:\EBrush\logs\session_C.brlog"
& $RC -m rcstudio project ebrush train --wait
```

从 E-Brush 移交清单导入，RC Studio 会**验证源文件大小与 SHA-256**，然后执行离线解析；源文件内容不修改。ZIP 文件内只读取 `.brlog` 成员，不解压整个工程，也不执行其中代码。对于 CSV，必须包含：

- 物理量列 `gx_rad_s`, `gy_rad_s`, `gz_rad_s`, `ax_m_s2`, `ay_m_s2`, `az_m_s2`；或
- 设备码列 `gx_raw`, `gy_raw`, `gz_raw`, `ax_raw`, `ay_raw`, `az_raw`。

可选 `sample_time_us` 或 `timestamp_us`，不提供时按名义 200 Hz 索引构造；**没有明确单位的 `gx, gy, ...` 不能自动猜测**。

所有采集文件都应有真实、稳定的采样时间关系。单个长录制即使切成 12 个片段，仍只算**一个独立采集记录**；训练端强制要求 **至少 3 个独立原始文件内容**。只上传一份真实日志时可以导入、检查和可视化训练样本，但**不会伪造独立泛化测试**。

### 图形界面操作

打开 GUI，在左侧的“研究项目”切换“通用 ESN” / “EBrush Motion Predictor V1”。EBrush 页面可以导入数据，查看片段、勾选会话、生成演示样本以及提交训练。右侧统一展示 **测试 RMSE / 持续性基线 RMSE / 预测步长 / 独立录制数**、时间序列曲线、神经元状态热力图及实验列表。浅色/深色模式使用 QSettings 持久化（Material 3 启发的原生 Qt 设计，无外部资源）。

## 3. EBrush Motion Predictor V1 的严格研究定义

| 项目 | V1 范围 |
|---|---|
| 输入 | IMU 陀螺仪三轴 rad/s + 加速度计三轴 m/s²，共 **6 通道** |
| 输出 | **未来角速度模长** `||gyro(t+10)||`，单位 rad/s |
| 预测 | 名义 200 Hz 采样率，`horizon=10` → **50 ms** |
| 模型 | 固定稀疏储备池 ESN + 岭回归读出层，默认 96 neurons |
| 评估 | 数据来源文件级隔离、训练统计量归一化、分片 washout、R²/MAE/RMSE、persistence 基线 |
| 预测模式 | 使用已观测输入的条件预测，依据实时观测进行前向预测；自主递归预测属于另一类设置 |
| 保存 | `model.npz`, `traces.npz`, `predictions.csv`, `summary.json`，SQLite 实验索引 |

训练时同一个记录文件的所有有效连续片段必须属于同一组。可通过 `project ebrush train --config examples/ebrush-motion.json --wait` 修改模型参数；`dataset.session_ids` 可列出要使用的片段 ID，为 `[]` 表示所有可用 EBrush 会话（注意此时不能混合演示与真实数据）。

`.brlog` 只使用 IMU 数据与 `device_status` 有效位：CRC、COBS、帧号和 MCU 时间戳用于质量分析；**不**把 Hall 的过期值、未标定视觉坐标或相机数据作为监督标签。实际时间基准是按连续 MCU 时间戳筛选后的名义采样率；跨记录物理测量一致性仍需设备端校准确认。

当前基线预测角速度标量。后续可以在 RC Studio 中扩展多目标读出与真实轨迹标签，用于笔尖位置、完整姿态或笔触等新任务。

## 4. 通用 ESN 保持兼容

```powershell
& $RC -m rcstudio check
& $RC -m rcstudio status
& $RC -m rcstudio sample-config --output config.json
& $RC -m rcstudio run --config .\examples\sine.json --wait
& $RC -m rcstudio runs
& $RC -m rcstudio show <run_id>
& $RC -m rcstudio traces <run_id> --limit 200 --units 16
& $RC -m rcstudio stop <run_id>
```

单变量 CSV 使用 `import-csv`，可在 GUI 的数据列选择器明确指定列名；此前 UI 在使用 CSV 时出现的未定义控件问题已修复。旧版工作区的 SQLite 表结构没有被迁移/破坏，新版本能打开旧实验记录；但不同版本的运行引擎和 GUI 可视化能力不应被混淆。

## 5. Agent CLI / HTTP / MCP

RC Studio 后台使用 `http://127.0.0.1:8766/v1`，仅监听 loopback。所有非 `/v1/health` 接口需要 Bearer Token：令牌在 `RCSTUDIO_HOME/credentials.json`，写权限与只读权限分开。**不能把端口直接暴露到公网**。

新旧接口：

- **只读 GET**：`/v1/health`, `/v1/workspace`, `/v1/projects`, `/v1/projects/ebrush/sessions`, `/v1/datasets`, `/v1/runs`, `/v1/runs/<id>`, `/v1/runs/<id>/traces`, `/v1/diagnostics`。
- **写入 POST（需要写令牌）**：`/v1/runs`, `/v1/projects/ebrush/train`, `/v1/runs/<id>/stop`。
- **不会提供**：相机控制、串口访问、E-Brush 标定、自动读取任意远程 URL、任意路径写 API。

MCP v2 配置示例（路径改成你自己的 Windows Python 绝对路径）：

```json
{
  "mcpServers": {
    "rcstudio": {
      "command": "C:\\RCStudio\\.venv\\Scripts\\python.exe",
      "args": ["-m", "rcstudio", "mcp"]
    }
  }
}
```

默认提供 `get_workspace`, `list_experiments`, `get_experiment`, `get_reservoir_states`, `get_diagnostics`, `list_research_projects`, `list_ebrush_sessions` 等**只读工具**。只有显式改为 `"args": ["-m","rcstudio","mcp","--allow-write"]` 才增加 `create_experiment`, `train_ebrush_motion`, `stop_experiment`。RC Studio 本身**没有集成任何 LLM Agent**。

> 关于 SDK：本版 MCP 采用 `mcp.server.mcpserver.MCPServer`（MCP Python SDK 2.x），对应 SDK 2.x 的导入路径；仍需在安装 MCP 的目标环境完成端到端连接测试。

## 6. 测试与限制

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

覆盖通用模型可复现性、CSV、SQLite 句柄释放、服务权限、EBrush 帧解析/CRC、移交 SHA-256、ZIP 安全导入、独立采集隔离、演示六通道训练、真实与模拟数据混合禁用、HTTP 只读/写令牌行为。

目前尚未完成：Windows 真机 GUI 验收、MCP 第三方客户端协议联调、多输入新录制的独立在线推理接口、闭环自由运行预测、自动超参数优化、笔尖轨迹真值校准、长期硬件可靠性验证。这些工作不应被自动测试通过所替代。

**许可提示：上传源码包的 `LICENSE` 实际为 GNU GPL v3，尽管 V0.1 README 曾写成 MIT。本版保留原 `LICENSE`，并纠正此处的说明。** 对外再分发请核实原项目版权信息并遵守 GPLv3。
