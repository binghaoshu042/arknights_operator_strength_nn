---
title: ARK
emoji: 🏆
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.0.0
python_version: '3.10'
app_file: app.py
pinned: false
short_description: what is your cup
---

# 🧠 测测你是什么杯 | 明日方舟六星干员战术神经网络评估仪 v3
> **Arknights Operator Strength Evaluation Neural Network System v3 (Unified Single Model & Dynamic Game-Mode Diagnostics)**

这是一个基于 **PyTorch 多层感知机 (MLP)** 神经网络的硬核明日方舟六星干员战术评估与杯级推演系统。项目采用极简轻奢的**北欧奶油色彩美学**，专为移动端及手机浏览器完美适配，并融合了**输入层梯度反向传播归因 (Input * Gradient Saliency)**，动态计算每个战术数值对干员强度的真实决策贡献！

---

## 🌟 核心硬核亮点

### 1. 🌐 单一大模型大一统 (Unified Neural Network)
* 摒弃了多模型相互割裂的结构，采用**单一大模型 (hidden dimensions: [256, 128, 64])** 对全部八大职业数据进行大一统训练。
* 将 `职业大类 (profession)`、`属性类型 (type)`、`细分分支 (branch)` 作为 categorical 输入联合一热编码（One-Hot）成 68 维特征，交由网络统一收敛。
* **弱化血统论**：细分分支对强度的自动得分占比限制在 **$2\%$ 以内**。模型被强迫忽略“出生光环”，仅根据干员实打实的 10 维实战数据给出推演结论！

### 2. ⚡ “数值极高即有就业空间”保底 (Extreme Niche Override)
* **数据层保底**：任何干员只要在 **DPS、DPH、技能总伤、控制覆盖、属性辅助、极限防御（重装）、点火速度（先锋）、SP 消耗效率（SP 越低效率越高）** 的任意一维战术参数达到同职业顶尖水平 ($\ge 85\%$)，自动上调模拟评分，使其保底为 **【大杯 (A-Tier)】** 或 **【超大杯 (S-Tier)】**。
* **接口层破格升档**：在推理服务器（`server.py`）中内置硬性升档规则——**只要出现任何明显高于正常均值的极致数据（如 ATK/DEF/DPS/DPH/总伤 高出均值 30% 以上，CC 高出 0.15 以上，或 SP 消耗低至 70% 以下）**，评定直接**无条件上升一个档次**！同时自动完成概率分布和梯度解释器的同步重映射，实现可信度完全闭环。

### 3. 🗺️ 深度游戏模式强度评估 (Dynamic Game-Mode Diagnostics)
基于用户调配的 10 维数据，前端诊断报告能够实时捕获并输出高度拟真的明日方舟硬核强度剖析：
* **优势区间 (Dominant Niche)**：解算干员属于极限爆发输出、钢铁承伤物理抗压、长时间控制抑或是高频运转。
* **最佳就业空间 (Employment Space)**：定位干员是决战核心 Main Carry、战术拖延辅控、钢铁阵线铁壁 Tank 还是极速启动 DP 回复核心。
* **探索者集成战略 (IS / 肉鸽模式)**：评估招募性价比、藏品适配度与中后期发育承伤表现。
* **危机合约高难 (CC / 挂词条模式)**：推演在 SP 恢复减缓、部署人数限制、敌方攻防倍增等恶劣限制 Tag 下的逃课或抗压底牌。

### 4. 🛡️ 严格单调性与法术 DPH 屏蔽
* **防御与法抗单调性**：在特征工程和数据生成中加入了绝对正向约束，确保增加 DEF 与 RES 只会提升或维持杯级，绝不出现“越肉杯级越低”的 MLP 负向拟合缺陷。
* **法术伤害干员 DPH 屏蔽**：由于法术伤害按百分比结算抗性，系统在底层对法术攻击干员的 DPH 进行自动归零屏蔽，强迫其 DPH 归一化输入为 0.00，实现完美的机制还原。

---

## 📁 项目目录结构

```
arknights_operator_strength_nn/
├── app/
│   ├── static/                    # 静态资源
│   ├── templates/
│   │   └── index.html             # 奶油极简 Mobile-First steppers 网页
│   ├── main.py                    # (可选) Streamlit 版调试入口
│   ├── server.py                  # Flask 推理服务器 (支持 OMP/PyTorch 单大模型加载)
│   └── utils.py                   # 职业均值、上限及 Streamlit 配套配置
├── data/
│   └── raw/
│       └── operators.csv          # 包含 840 条极值样本的高质量 6 星大集数据
├── models/
│   ├── unified_model.pth          # PyTorch 统一大模型权重文件 (Accuracy: ~60.1%)
│   └── unified_preprocessor.pkl   # 联合一热编码与 numerical 标准化 preprocessor
├── src/
│   ├── data/
│   │   ├── data_loader.py         # 数据加载与 Train-Test 划分层
│   │   └── generate_mock_data.py  # 包含 SP 逆消耗及极值就业 override 模拟数据生成器
│   ├── evaluation/
│   │   └── evaluator.py           # 混淆矩阵生成与 Input * Gradient 决策梯度归因引擎
│   ├── features/
│   │   └── feature_extractor.py   # StandardScaler & OneHotEncoder 联合特征提取器
│   └── models/
│       └── nn_model.py            # 三层高容量 MLP 神经网络架构
├── train.py                       # 大一统模型训练控制入口 (全量数据 fit 并导出)
└── requirements.txt               # 基础运行依赖 (torch, pandas, flask, scikit-learn 等)
```

---

## 🚀 极速安装与启动指南

### 1. 准备环境 (建议使用 Anaconda/Miniconda)
```bash
# 创建并激活 conda 虚拟环境
conda create -n arknights_nn python=3.10 -y
conda activate arknights_nn

# 安装依赖
pip install -r requirements.txt
```

### 2. 重新训练大一统神经网络 (可选项)
如果您修改了数据生成系数，可以删除 operators.csv 并触发重新训练：
```bash
# 删除旧缓存
Remove-Item data\raw\operators.csv -ErrorAction SilentlyContinue

# 执行大一统重训 (会自动生成 operators.csv 并对 68 维特征网络训练收敛)
python train.py
```

### 3. 启动 Flask 战术服务器
```bash
python app/server.py
```
启动成功后，服务器会执行 `Pre-loading unified neural network model...` 并加载唯一大模型，监听本地端口：
`http://127.0.0.1:5000`

### 4. 浏览器体验
用手机浏览器或电脑浏览器打开 `http://127.0.0.1:5000`，体验最纯粹的明日方舟神经网络测试网页 **“测测你是什么杯”** 吧！

---

## 🔬 神经网络 API 说明 (`/api/predict`)

系统提供标准 REST API 支持实时推演，接收干员基础属性 JSON 并返回杯级推演与归因矩阵：

### 1. 请求示例 (POST)
* **URL**: `http://127.0.0.1:5000/api/predict`
* **Headers**: `Content-Type: application/json`
* **Body**:
```json
{
    "name": "蕾缪安",
    "profession": "Sniper",
    "branch": "战术射手",
    "type": "物理",
    "atk": 1500,
    "def": 200,
    "res": 5,
    "dps": 4500,
    "dph": 14000,
    "total_damage": 80000,
    "sp_cost": 45,
    "init_sp": 25,
    "control_coverage": 0.0,
    "buff_amp": 1.0
}
```

### 2. 响应示例 (JSON)
```json
{
  "success": true,
  "operator_name": "蕾缪安",
  "predicted_cup": "超大杯",
  "confidence": 0.8638,
  "probabilities": {
    "小杯": 0.000018,
    "中杯": 0.00012,
    "大杯": 0.1360,
    "超大杯": 0.8638
  },
  "attributions": [
    { "feature": "单次常规物理攻击力 (DPH)", "value": 4.0723 },
    { "feature": "技能总伤 (Total Dmg)", "value": 2.3045 },
    { "feature": "辅助增伤幅度 (Buff)", "value": -1.0831 }
  ],
  "averages": { "atk": 950, "dps": 1400, "dph": 1500, "total_damage": 60000 },
  "limits": { "atk": [400, 2200], "dph": [300, 15000], "total_damage": [10000, 350000] }
}
```

---

## 🎨 奶油极简 Mobile-First 视觉风格展示
项目没有使用任何臃肿的游戏大图和杂乱的暗色调，而是采用了**纯净明亮的暖色系配色方案**：
* 🎨 北欧奶油灰背景 (`#f7f9fc`) 结合质感纯白玻璃卡片 (`#ffffff`)；
* 🌾 优雅防尘金 (`#d4a373`) 与复古深青蓝 (`#5c868d`) 双重渐变；
* 📊 针对不同杯级匹配了治愈系温暖红 (`#e07a5f`) 与鼠尾草绿 (`#8fa89b`) 边框；
* 📱 精心解构的五道极速分步表单，滑块反馈丝滑，Chart.js 动态绘制偏离图与决策贡献，手机单手握持操作极其舒适！
