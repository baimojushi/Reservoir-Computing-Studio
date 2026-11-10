# RC Studio v0.1 — 储备池计算研究工作台

Windows 本地开源原型：**真实回声状态网络（ESN）训练 + 图形界面 + PowerShell CLI + 本地 TCP/HTTP 服务 + MCP 标准输入输出工具**。

> 交付状态：代码与自动测试已在当前环境执行；正式 Windows / PySide6 GUI 与 MCP 客户端兼容性需要在目标笔记本上验收。此原型使用经过测试的 NumPy/SciPy ESN 引擎，尚未接入 ReservoirPy。CLI/GUI/MCP 使用统一后台服务。

## 1. 系统要求

- Windows 10/11，64 位，Python 3.11 或更新版本。
- 普通 CPU；默认 120 个储备池神经元，不要求独立 GPU。
- 首次安装 PySide6、SciPy 和可选 MCP SDK 时需要访问 Python 包源。
- GUI 为可选依赖；无 GUI 时仍可安装并执行 CLI 与服务器。

## 2. Windows 快速启动

解压工程后，在 PowerShell 中进入项目文件夹：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

`start.ps1` 自动启动本地 HTTP 计算服务，并打开 GUI。
如需测试：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

## 3. CLI / PowerShell

通过虚拟环境的 Python 调用（不依赖 PATH）：

```powershell
$RC = ".\.venv\Scripts\python.exe"
& $RC -m rcstudio check
& $RC -m rcstudio status
& $RC -m rcstudio sample-config --output config.json
& $RC -m rcstudio run --config .\examples\sine.json --wait
& $RC -m rcstudio runs
& $RC -m rcstudio show <实验ID>
& $RC -m rcstudio traces <实验ID> --limit 100 --units 12
& $RC -m rcstudio stop <实验ID>
```

`run` 返回 16 位十六进制实验 ID，所有 CLI 响应是 JSON，方便外部智能体解析。
`--wait` 仅等待 CLI 进程；训练始终由本地服务后台执行。
`run --wait --timeout 300` 达到超时时，仅终止等待，后台任务仍继续。

### 自定义 CSV 数据

```powershell
& $RC -m rcstudio import-csv "D:\Research\signals.csv"
& $RC -m rcstudio datasets
```

复制返回的 `id`，在配置文件中使用：

```json
{
  "dataset": {
    "kind": "csv",
    "dataset_id": "复制这里的32位数据集ID",
    "column": "signal"
  },
  "model": {"units": 100, "horizon": 1}
}
```

如果不指定列，自动选择 CSV 中第一个全部为有限数值的列。数据必须带表头。
V0.1 限制 CSV 大小 20 MB、长度 20,000 行；支持单变量时间序列。
CSV 文件导入后存放在工作台管理目录。

### 使用保存的模型预测新 CSV

```powershell
& $RC -m rcstudio predict <已完成实验ID> --dataset-id <数据集ID> --column signal
```

该命令加载已训练权重，计算对应预测步长的**观测输入预测**，不重新训练。
使用 `--output` 可指定预测 CSV 的导出路径。

## 4. MCP 接口：外部 LLM Agent

RC Studio 不嵌入 Agent 或 LLM。MCP SDK v2 使用标准输入输出协议。
典型客户端配置（请将 `command` 改为本机 Python 的**绝对路径**）：

```json
{
  "mcpServers": {
    "rcstudio": {
      "command": "D:\\RCStudio-v0.1\\.venv\\Scripts\\python.exe",
      "args": ["-m", "rcstudio", "mcp"]
    }
  }
}
```

MCP 默认只读工具：
`get_workspace`, `list_experiments`, `get_experiment`,
`get_reservoir_states`, `get_diagnostics`。

如确认允许 Agent 提交与停止训练，配置为：

```json
"args": ["-m", "rcstudio", "mcp", "--allow-write"]
```

增加 `create_experiment`、`stop_experiment` 两个写工具。建议首次只开放只读。
MCP 连接会自动启动本机后台服务（如未运行）。

**数据策略：** 外部云端 LLM 收到多少数据由 MCP 客户端及其授权控制；
请勿将含敏感信息的 CSV 或实验日志发送到云端。MCP 默认仅提供摘要和受限状态采样。

## 5. 本地 HTTP / TCP

默认：`http://127.0.0.1:8765/v1`，端口可通过环境变量设置：

```powershell
$env:RCSTUDIO_PORT = "8766"
& $RC -m rcstudio serve --port 8766
```

API：`GET /v1/health`、`GET /v1/workspace`、`GET /v1/datasets`、
`GET /v1/runs`、`GET /v1/runs/{id}`、`GET /v1/runs/{id}/traces`、
`GET /v1/diagnostics`、`POST /v1/runs`、`POST /v1/runs/{id}/stop`。

除健康检查外，API 必须提供 `Authorization: Bearer <token>`。
令牌存储在当前用户的 `credentials.json`，分为只读和写入能力。
服务仅监听 `127.0.0.1`，并拒绝非本机 Host 和浏览器 Origin。
**禁止将本地端口直接转发到公网。**

默认工作目录：
- Windows：`%LOCALAPPDATA%\RCStudio`
- Linux/macOS：`~/.local/share/rcstudio`
- 自定义：`RCSTUDIO_HOME` 环境变量。

每次启动服务后，未完成的旧任务标记为 `interrupted`，不会悄悄继续运行。

## 6. 模型数学与实验定义

首版是固定循环权重的回声状态网络（ESN），读出层使用岭回归：
`s(t)=(1-leak)*s(t-1)+leak*tanh(W*s(t-1)+Win*x(t)+b)`。

- 输入：一条实数时间序列；默认合成信号。
- 预测：时刻 `t` 的历史状态预测 `t+h` 的目标；验证与测试采用真实观测输入。
- 默认时间顺序划分：训练 70% / 验证 15% / 测试 15%。
- 标准化仅使用训练段统计量，训练阶段前 `washout` 步被跳过。
- 指标：RMSE、MAE、R²、持续性基线的测试误差。
- 固定随机种子，保存 `model.npz`, `traces.npz`, `predictions.csv`, `summary.json`。
- GUI 预测图及状态热力图为真实计算结果，状态采样是下采样展示。

**特别注意：** 这属于有真实新观测输入的条件预测（teacher-forced /
observed-input forecasting），尚未实现自主递归多步外推，且不等同于闭环预测。
谱半径基于初始化循环权重的最大特征值模进行缩放。

## 7. 常见问题

- `ModuleNotFoundError: PySide6`：安装 GUI 依赖 `pip install -e ".[gui]"`。
- `ModuleNotFoundError: mcp`：安装 `pip install -e ".[mcp]"`。
- 端口被其他进程占用：指定 `RCSTUDIO_PORT` 并保持客户端一致。
- GUI 不显示数据：确认实验状态为 `finished`，双击实验行。
- PowerShell 脚本无法执行：使用上述 `-ExecutionPolicy Bypass -File` 命令，仅对该进程生效。
- 原始 CSV 不正确：确保有表头，数值列无缺失或非有限值。
- 实验停止：任务逐时间步检查停止标志；大型线性求解期间可能稍有延迟。

## 8. 已知限制与后续计划

V0.1 优先确保单储备池 ESN 全流程可用：
- 网络结构图是只读示意图，暂无拖放式任意拓扑编辑。
- 尚无 LSTM/GRU 训练器、在线训练、多储备池网络与外部设备连接。
- HTTP JSON 面向本机使用，MCP 当前采用 stdio，暂未提供 Streamable HTTP。
- 仅实现有限长度单变量 CSV、基础状态热力图，尚无高级谱/记忆容量分析。
- 内部训练器为 NumPy/SciPy ESN，ReservoirPy 可在下个版本作为替换式适配器。
- Windows GUI 及第三方 MCP 客户端尚需目标机器测试，视依赖版本可能需要调试。

本项目源代码采用 MIT 许可证；第三方依赖各自遵循其许可证。
