# RC Studio V0.7.2 · 主工程融合修复与验收说明

> 基线：用户提供的 `RC-Studio-V0_7_1-merged-main.zip`（V0.7.1）。本次以该主工程为直接修改对象，保留训练引擎、离线 EBrush 数据集、CLI、HTTP、MCP、数据库及已有五步实验室。
>
> 结论范围：NumPy/SciPy、服务契约、JavaScript 计算、系统 Chromium 实测已在本次环境运行；**PySide6 缺失，Windows 原生 Qt/QtWebEngine 窗口仍待实机验收**。

## 1. 解决的 P0 问题：独立录制不能拼成一条计算因果链

之前 EBrush 在每一份 session 开始时重置储备池状态，但轨迹文件的 `timestep` 在不同录制间连续递增。浏览器只按全局 `index+h` 平移预测，因此将上一段录制的预测延伸到下一段的预热区间，造成错误的时空连续关系。

现在保持模型训练和得分计算过程不变，增加完整的**轨迹身份契约**：

| 层 | 文件 | 新增约束 |
|---|---|---|
| 训练/保存 | `rcstudio/ebrush.py` | 每个状态帧保存 `episode_id`、`episode_start`、`local_step`、`target_local_step`；附 episode→recording/session 的映射 |
| 只读 API | `rcstudio/core.py` | `reservoir_frames` 在窗口内输出相同元数据及 `has_episode_metadata`；不会猜测旧版文件的录制分割 |
| 统一数值 | `rcstudio/episode_time.py` | `align_to_target`: 用 `(episode_id, target_local_step)` 寻找输出目标帧，不在不同录制间移动预测；未采样帧、窗口外和 washout 均保留 NaN |
| Web | `rcstudio/lab_web/lab.html` | 预测按录制内目标时刻对齐，输入/状态/输出曲线在边界断开；显示 `新录制·状态重置`，播放到边界暂停 |
| Qt 回退 | `rcstudio/intuitive_view.py` | 与 Web 采用同一录制局部时间约束；跨边界状态不插值；绘图断线；时间标签明确片段身份；播放暂停 |

**时间数学定义**：在某独立录制 e 的输入步 s 计算状态 `x_e(s)`，读出预测对应 `y_e(s+h)`；只允许匹配 `(e,s+h)`。不能将该值放在下一个录制 `(e+1,0)`，即使它们在拼接数组中相邻。

**历史数据的安全降级**：旧 EBrush `traces.npz` 没有录制边界元数据时，不凭状态突变猜测边界。Web 和 Qt 只展示可以确认的神经元状态；隐藏输入/输出连续曲线与预测读出，并提示使用新版重新训练。通用单时间序列 ESN 不受影响。

## 2. P1 交互与教学修复

- **Qt 原生回退可做课程**：增加“敲一下”“选择节点”“查看训练读出”“调整参数”的实际操作入口，并发出与课程适配器一致的 `chapter_completed` 事件。用户不必靠跳过课程 00 才能推进；“查看读出”是查看已求解的示范读出，不伪称点击后才开始训练。
- **手绘反方向插值**：加入独立的 `sketch_math.interpolate_stroke`，以两个实际端点的索引/值求线性插值。自右向左和自左向右得到相同波形。
- **QtWebEngine 回退更透明**：只捕获预期的导入、系统或运行期加载失败（`ImportError`、`OSError`、`RuntimeError`），日志输出异常堆栈，回退界面显示失败原因。其他程序错误直接暴露，不会被假装成正常降级。
- **Windows 发布测试**：`scripts/test.ps1` 同时执行 `tests` 与 `learning_tests`，避免只测计算、不测教学。
- **浏览器旧实验空值兼容**：在旧轨迹禁用读出之后，页面说明也能处理 `parts=null`；空值预测通过公开 probe API 返回 JSON 的 `null`，不再触发 JavaScript TypeError。
- **离线 Chromium 走查**：`scripts/check_lab_page.py` 优先使用系统 Chromium，适配禁止 `file://` 导航的沙箱；新增 `scripts/check_episode_browser.py` 使用真实 `train_motion` 结果验收浏览器的录制边界和旧数据降级。

## 3. 自动化回归与可复核证据

1. `python -m pytest -q tests learning_tests`：**155 passed, 1 skipped, 6 subtests passed**。
2. `python -m rcstudio check`、`python -m rcstudio --version`：通过，版本 `0.7.2`。
3. `python scripts/check_lab_page.py <输出文件夹>`：8 个页面场景，0 个问题。
4. `python scripts/check_episode_browser.py <输出文件夹>`：5 份合成录制由正式 EBrush 引擎训练，窗口跨越两处录制边界，9 项 Web 验收通过、0 个 JavaScript 异常。可核对 JSON、PNG 和 `tests/test_episode_contract_v072.py`。
5. 离线构建 wheel；SHA-256 源码清单重新生成。

**说明**：上述 5 份录制是 `make_demo()` 合成输入，训练和浏览器页面使用的是真实工程代码及实际计算输出，但不是人体动作测量，不代表真实电子毛笔的预测精度。

## 4. 发布前必须完成的原生平台验收

- [ ] Windows 10/11 启动 GUI、嵌入 QtWebEngine；切换“认识 ESN/研究工作台”不丢失实验、状态与帮助绑定。
- [ ] 卸载/禁用 QtWebEngine 后实际进入原生回退；观察失败原因；依次完成课程 00 的四项操作。
- [ ] 在原生回退用鼠标从右向左、从左向右手绘，检查两端状态及实时更新。
- [ ] 用真实电子毛笔独立录制数据训练：验证每段局部时间、状态重置、洗去阶段、h 步目标和输出单位的对应。
- [ ] 在内嵌 QtWebEngine 加载有多个录制边界的 EBrush 实验，验证播放到边界暂停和逐段预热空白。
- [ ] 大屏/HiDPI、窗口伸缩、加载取消、线程清理、异常反馈与大规模储备池帧率。

## 5. 完整工程入口

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1
```

仅计算/测试（不依赖 PySide6）：

```bash
python -m pip install -e .
python -m rcstudio check
python -m pytest -q tests learning_tests
```

浏览器专用走查需额外安装 `playwright` 且存在 Chromium，可选，不影响生产应用与数值引擎的安装。

## 6. 验收判定

**通过**：主工程计算及教学回归、真实 EBrush 合成数据的录制时间契约、旧数据安全降级和 Chromium 页面专项。**未完成**：Windows 原生 QtWebEngine 与原生回退的实机视觉/交互验收；真实电子毛笔硬件与人体运动数据验证。此次改版是 V0.7.1 → **V0.7.2** 的增量修复，不改变训练算法或模型参数含义。
