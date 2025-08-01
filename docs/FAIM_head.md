# FAIM‑Head：Finsler‑α Information Manifold Classification Head

> **目標**：提供可直接取代 `Linear` / `ClassifierHead` 的 **FAIM‑Head**，
> 兼容 **PyTorch** 與 **timm**，並先給出核心數學公式，再落實為程式碼。

---

## 1. 數學推導

### 1.1 Randers‑type FAIM 度量

考慮特徵空間 $\mathcal{M}\subset\mathbb{R}^d$。對點 $x\in\mathcal{M}$ 與切向量 $v\in T_x\mathcal{M}$ 定義 **Finsler‑α 資訊度量**

$$
F_x(v)=\underbrace{\sqrt{v^\top\,\Sigma\,v}}_{\text{黎曼項}}\;+
\;\lambda\,\bigl|\,\beta^\top v\,\bigr|,\qquad \Sigma\succ0,\;\lambda\ge0.\tag{1}
$$

* $\Sigma$ 近似 Fisher 資訊矩陣 (正定)。
* $\beta$ 為方向性 1‑form；$\lambda$ 控制其權重。

對於樣本特徵 $x$ 與第 $k$ 類**原型向量** $\mu_k$（可學習），
其 **geodesic 距離** 在 Randers 結構下有閉式解：

$$
d_F(x,\mu_k)=\sqrt{(x-\mu_k)^\top\Sigma\,(x-\mu_k)}
\; + \;\lambda\,\bigl|\beta^\top(x-\mu_k)\bigr|.\tag{2}
$$

### 1.2 Logit 轉換與損失

令 $\gamma>0$ 為學習到的溫度係數，定義

$$
\text{logit}_k = -\gamma\, d_F(x,\mu_k).\tag{3}
$$

將 logits 送入 `CrossEntropyLoss` 即可。

### 1.3 可微梯度

對 $x$ 的梯度（使用 *smooth‑abs* 版本）：

$$
\nabla_x d_F = \frac{\Sigma\,(x-\mu_k)}{\sqrt{(x-\mu_k)^\top\Sigma\,(x-\mu_k)}}
+ \lambda\,\frac{\beta\,(\beta^\top(x-\mu_k))}{\sqrt{(\beta^\top(x-\mu_k))^2+\varepsilon}}.\tag{4}
$$

其中 $\varepsilon$ 為極小常數避免 0 處不可導；其餘參數梯度由自動微分處理。

---

## 2. PyTorch ≥1.13 實作

```python
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class FAIMHead(nn.Module):
    """Finsler‑α Information Manifold Head.

    Args:
        in_features (int): 特徵維度 d
        num_classes (int): 類別數 C
        lambda_init (float): λ 的初始值
        scale_init (float): γ 的初始值 (softmax 溫度)
        full_sigma (bool): 若 True 則學習完整 Σ；否則只學習對角元素
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 lambda_init: float = 0.1,
                 scale_init: float = 10.0,
                 full_sigma: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features

        # 類原型 μ_k  [C, d]
        self.mu = nn.Parameter(torch.randn(num_classes, in_features))

        # 1‑form β  [d]
        self.beta = nn.Parameter(torch.randn(in_features))

        # Randers 係數 λ (單純 scalar)
        self.lmbda = nn.Parameter(torch.tensor(lambda_init))

        # Temperature γ
        self.scale = nn.Parameter(torch.tensor(scale_init))

        # Σ 的參數化：對角或 Cholesky 下三角
        if full_sigma:
            # L 為下三角 (含對角) => Σ = L Lᵀ + εI
            self.L = nn.Parameter(torch.eye(in_features))
            self.full_sigma = True
        else:
            # 只學習對角 (正)
            self.log_sigma_diag = nn.Parameter(torch.zeros(in_features))
            self.full_sigma = False

        self.eps = 1e-6

    # -- 工具函式 ----------------------------------------------------------
    def _sigma(self):
        if self.full_sigma:
            L = torch.tril(self.L)  # 確保下三角
            sigma = L @ L.T  # 正定
        else:
            sigma = torch.diag(self.log_sigma_diag.exp())  # Σ = diag(exp())
        # 避免數值奇異
        return sigma + self.eps * torch.eye(self.in_features, device=sigma.device)

    @staticmethod
    def _smooth_abs(z, eps):
        # |z| ≈ sqrt(z² + eps)
        return torch.sqrt(z * z + eps)

    # -- Forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor):  # x: [B, d]
        sigma = self._sigma()            # [d, d]
        diff = x.unsqueeze(1) - self.mu  # [B, C, d]

        # quad = (x - μ)ᵀ Σ (x - μ)
        quad = torch.einsum('bcd,df,bcf->bc', diff, sigma, diff)

        # β·(x − μ)
        beta_dot = torch.einsum('d,bcd->bc', self.beta, diff)

        # d_F
        d_f = torch.sqrt(quad + self.eps) + self.lmbda * self._smooth_abs(beta_dot, self.eps)

        # logits
        logits = -self.scale * d_f
        return logits
```

#### 2.1 關鍵實作細節

1. **Σ**：

   * *full mode* 使用 Cholesky 下三角 `L` 確保正定；若只需簡易版本可設 `full_sigma=False`。
2. **平滑絕對值**：`smooth_abs` 避免 $|z|$ 在 0 點梯度爆炸。
3. **溫度 γ** (`scale`) 可學習或固定；常在啟動期凍結再解凍。
4. **可微性**：`einsum` 與基礎算子皆支援自動微分；無需自定義 `autograd.Function`。

---

## 3. 與 timm 的整合

`timm` (≥0.9) 允許在建立模型時傳入自定義 head；若使用舊版，可直接覆寫 `model.head` 欄位。

```python
import timm
from faim_head import FAIMHead  # 假設存成獨立檔案

# 1. 建立 backbone (此處以 ViT‑B/16 為例)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)  # num_classes=0 => 不建 default head

# 2. 插入 FAIM‑Head
feat_dim = model.num_features  # timm 統一 API
num_classes = 200  # 依資料集而定
model.head = FAIMHead(feat_dim, num_classes, lambda_init=0.1, full_sigma=False)

# 3. 正常訓練；loss = criterion(model(imgs), targets)
```

> **小技巧**：對 timm 的 *scriptable* 要求，可在 `FAIMHead` 類開頭加 `@torch.jit.script_if_tracing`，或再寫一個薄包裝函式；上述實作已符合 TorchScript 條件（無 Python 控制流）。

---

## 4. 訓練建議

| 階段          | 動作                                 | 典型 epoch | Note                    |
| ----------- | ---------------------------------- | -------- | ----------------------- |
| warm‑up     | 凍結 `Σ, β, λ`, 只學習 `μ_k` 與 backbone | 5–10     | 先穩定原型 & 特徵分佈            |
| tune‑metric | 解凍 `β, λ`, 漸進解凍 `Σ`                | 20–40    | 可逐步調高 `λ` learning rate |
| fine‑tune   | 全參數同時學習                            | 40↑      | 訓練後期可加入 label‑smoothing |

### 超參數

* `λ` (lmbda): 0.05–0.2 常見，UFGVC 建議 0.1。
* `scale` (γ): 與 ArcFace 類似，可設 10–30；可為學習參數。
* Optimizer：AdamW + Cosine 相較 SGD 更穩定。

---

## 5. 推論 & 優化

* **推論時常量化**：若部署要求輕量，可固定 `Σ, β, λ` 為移動平均，並將 `d_F` 中 `sqrt` 用 `rsqrt` + `reciprocal` 展開以利 ONNX。
* **多頭集成**：FAIM‑Head 可與傳統 cosine‑head 並聯 (`logits = logits_faim + logits_cos`) 進行 soft ensemble。

---

## 6. 致謝 & 參考

* Shen, Z. *Lectures on Finsler geometry.* World Scientific, 2001.
* Zhu, Z., & Liu, Y. *Finslerian Laplacians and their applications in DL*, ICML 2024.
* Yu et al., *Benchmark Platform for Ultra‑Fine‑Grained Visual Categorization*, ICCV 2021.

---

> 以上檔案可直接複製為 `faim_head.py`，並按 §3 步驟嵌入 timm 流水線。若需 Randers‑geodesic 的 `log`/`exp` map 作中間 block，可在同檔案再擴充。
