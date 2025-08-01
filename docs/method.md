### 在 CV 分類流程中使用 FAIM 的三種典型做法

| 做法                           | 整合位置                                          | 關鍵變動                                                                                                                                          | 適用情境                                     |
| ---------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **A. FAIM-Head**（最常用）        | **最後分類頭**<br>取代 `Linear(in_dim, num_classes)` | - 為每一類維護**原型向量** μ<sub>k</sub>  <br>- 以 **FAIM geodesic**  $d_F(x, μ_k)$ 轉成 logits:<br> `logits_k = −γ · d_F(x, μ_k)`<br>- 直接接 `CrossEntropy` | 想**零改動 backbone**，又要把流形特性反映到 Softmax 分界時 |
| **B. FAIM Contrastive Loss** | **損失函數**（InfoNCE／Triplet／ArcFace 等）           | - 把歐氏距離替換成 geodesic<br>- 不需額外參數，梯度自動傳回 backbone                                                                                               | 需要**度量學習**或小批量 N-pair 設計時                |
| **C. FAIM Block**（進階）        | **中間特徵層**（類似 Hyperbolic 層）                    | - `Log map → 𝑊 → Exp map` 形式<br>- 可堆疊多層，於低層放大 fine detail                                                                                    | 想在多層次逐步放大微差、或做特徵級對齊時                     |

---

#### A. FAIM-Head 參考實作（PyTorch pseudo-code）

```python
class FAIMHead(nn.Module):
    def __init__(self, feat_dim, n_cls, λ=0.1):
        super().__init__()
        self.mu     = nn.Parameter(torch.randn(n_cls, feat_dim))   # 類原型
        self.L      = nn.Parameter(torch.eye(feat_dim))            # Σ½ 確保正定
        self.beta   = nn.Parameter(torch.randn(feat_dim))          # 1-form β
        self.lmbda  = nn.Parameter(torch.tensor(λ))
        self.scale  = nn.Parameter(torch.tensor(10.0))             # γ

    def _sigma(self):
        # Σ = L Lᵀ + εI
        eye = torch.eye(self.L.size(0), device=self.L.device)
        return self.L @ self.L.T + 1e-6 * eye

    def forward(self, x):                     # x: [B, D]
        Σ   = self._sigma()                  # [D, D]
        diff = x.unsqueeze(1) - self.mu      # [B, C, D]
        quad = torch.einsum('bcd,df,bcf->bc', diff, Σ, diff)  # (q-p)ᵀ Σ (q-p)
        beta_dot = torch.einsum('d,bcd->bc', self.beta, diff) # β·(q-p)
        d_F = torch.sqrt(quad) + self.lmbda * beta_dot.abs()  # geodesic
        logits = -self.scale * d_F                            # 熱度 γ
        return logits
```

*特點*

1. **記憶體與算量**：與常規 `Linear`/CosFace head 同級，只多一個 `einsum`。
2. **參數安全**：`Σ` 以 `L` 下三角 → 自帶正定性，梯度無須額外投影。
3. **推論**：可選固定 `Σ, β` 為訓練期均值，或持續微調。

---

#### B. 在對比式損失中呼叫 FAIM 距離

```python
loss = F.cross_entropy(logits_pos / τ, torch.zeros(B, dtype=torch.long))
# logits_pos = -d_F(anchor, positive)
```

* 對 **Triplet**：`margin + d_F(anchor,pos) – d_F(anchor,neg)`。
* 對 **ArcFace**：把歐氏/餘弦 margin 的距離項換成 geodesic。
  *好處：* 不改網路圖，只改距離計算；對 **小 batch** 亦穩定。

---

#### C. FAIM Block（可選）

```text
x_in          # R^D feature
 │  (Log map : f ↦ v in T_pM)
v = log_F(p, x_in)
 │  (線性/注意力/ViT block on tangent)
v' = W(v)
 │  (Exp map : T_pM ↦ M)
x_out = exp_F(p, v')
```

* *Log/Exp* 在 **Randers 型 FAIM** 有閉式；各層仍能進梯度。
* 堆疊多層時，可用 **共享 p** 或 **動態 p=f(x)**（類似動態中心）。
* 實務：常在最後兩層才換成 FAIM block，以免前段過早擴張噪聲。

---

### 何時選何種整合？

| 訓練規模      | 資料性質      | 建議                     |
| --------- | --------- | ---------------------- |
| 少量資料 + 微調 | 全局特徵已夠    | **A**：最少改動，收斂快         |
| 需大量對比     | 有多檔增強     | **B**：提升度量表現           |
| 要全程幾何一致   | 雜訊極低但細節重要 | **A+B** 或 B + 少數 **C** |

---

### 反向可微與穩定性

* $d_F$ 由 **平凡函數**（`sqrt`, `abs`) 組成 → **PyTorch/LIB autograd** 天生支援。
* `abs` 在 0 點不可導，但機率極低；可用 `smooth_abs = sqrt(v² + ε)` 替代。
* 若擔心 **β 過大**，在訓練早期對 `|β·v|` 加 `sigmoid` 緩和。

---

### 與現有工作連結

* Finsler-LBO、Randers geodesic 已有可微實作，證實可嵌入神經網路 ([arXiv][1])。
* Graph 與對比式嵌入亦驗證 Finsler-Riemannian 空間優於純黎曼 ([Proceedings of Machine Learning Research][2])。

---

**摘要**

> *最快起步*：把最後 `Linear` 換成 **FAIM-Head**；若已有對比學習則另外加 **FAIM 距離**。
> *需要深入微調*：在高層插入 1–2 個 **FAIM Block**，讓特徵逐層在方向性曲率下變形。
> 所有方案都保持**可微同構**，不破壞原有 ViT／CNN 前端，且只需少量新增參數，即可在 UFGVC 場景擴大微差並穩定收斂。

[1]: https://arxiv.org/abs/2404.03999?utm_source=chatgpt.com "Finsler-Laplace-Beltrami Operators with Application to Shape Analysis"
[2]: https://proceedings.mlr.press/v139/lopez21a/lopez21a.pdf?utm_source=chatgpt.com "[PDF] Symmetric Spaces for Graph Embeddings: A Finsler-Riemannian ..."
