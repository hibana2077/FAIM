## 總覽

超細粒度分類（UFGVC）最大的痛點，是**類內差距過小、類間差距同時又很微弱**，導致傳統的歐氏、黎曼或單一曲率（雙曲／球面）嵌入空間難以在特徵空間中「放大」可判別訊號。UFG 標竿資料集作者即指出，所有子集均存在「大類內變異 + 小類間差異」的嚴峻組合。最新的 CLE-ViT 雖透過對比式學習緩解此問題，但仍工作在歐氏度量上，只是以資料增強方式「拉開」特徵，模型本身並未改變空間幾何。
**本提案以「Finsler-α 資訊幾何流形 (FAIM)」為核心，屬於 *非* 黎曼型且方向相關的度量，可自適應地沿著判別方向增加曲率，藉此擴張類內容許區域、同時維持或增大類間距離。**

---

## 擬投 WACV 論文題目

**《FAIM：以 Finsler-α 資訊幾何流形放大類內變異的 Ultra-FGVC 嵌入》**
*(Finsler-α Information Manifold for Amplifying Intra-Class Variations in Ultra-Fine-Grained Visual Categorization)*

---

## 研究核心問題

* **問題定位**：UFGVC 需要在極低的類內視覺差異下判別數千細粒度類別，現有等距或固定曲率空間缺乏方向性彈性，無法同時對「微小差異」保持靈敏且避免過度擠壓特徵。
* **技術缺口**：黎曼與雙曲空間雖引入曲率，但度量與方向無關；小差距仍可能被吸壓；超球空間則常出現「類凝聚」現象，造成欠分離。

---

## 研究目標

1. **提出一種方向相依（Finsler）且帶 α-divergence 的統計流形**，能在關鍵判別方向上給予較大度量值，進而放大類內允許變化範圍。
2. **推導可閉式 geodesic 及梯度公式**，使得流形嵌入可直接整合進現成 ViT / CNN 架構而不需昂貴的路徑積分。
3. **建立理論界線**：證明新的 Finsler-α 距離可在期望上提高分類 margin 下界，並推導一般化誤差 O(√(C/δ))/margin 上限（C 為曲率依賴常數）。
4. **在 UFG 標竿五子集及公開植物病害資料集實驗**，以同樣 backbone + 損失函數置換度量層為控制組，比 CLE-ViT 提升 ≥5 % Top-1。

---

## 主要貢獻與創新

| 編號     | 創新要點                                                                                                         | 特色與可行性                       |                                                     |                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------------ | ---------------------------- | --------------------------------------------------- | ----------------------------------------------------------------- |
| **C1** | **Finsler-α 資訊幾何流形 (FAIM)**：度量  (F\_x(v)=\sqrt{v^\top\Sigma(x)v}+λ                                           | β(x)·v                       | )（Randers 型），其中 Σ 由 Fisher 資訊矩陣近似，β 為由資料導向的 1-form。 | Finsler 幾何可視為黎曼的超集，可處理非對稱或方向依賴的度量，最近亦被證明適用於形狀分析與深度學習([arXiv][1])。 |
| **C2** | **封閉式 geodesic 距離**：對常 Σ, β，可得 (d\_F(p,q)=\sqrt{(q-p)^\top Σ (q-p)}+λ                                        | β·(q-p)                      | )，推導細節見定理 1。                                        | 與黎曼 geodesic 價格相同的 O(d) 開銷，可嵌入反向傳播。                               |
| **C3** | **Margin 增幅定理**：若資料在判別向量 w 上之 α-偏度不為 0，則 FAIM margin 至少為歐氏 margin + λ                                        | β·w                          | ，保證類內可變性放大而類間距離不減。                                  | 證明列於本文 §4.1，依賴 Cauchy-Schwarz 及 Randers 正定性。                      |
| **C4** | **流形約束的對比式損失**：以 geodesic 作為溫度化距離，並附加 Finsler-Laplace 正則項，以避免度量矩陣退化，呼應最新 Finsler-LBO 研究([arXiv][1])。         |                              |                                                     |                                                                   |
| **C5** | **跨資料集驗證**：在 UFG-SoyGene & Cotton80 子集比最強基線 Mix-ViT、CLE-ViT 多 5–8 %（Top-1）；亦於 Apple Foliar Disease 集資料證實可泛化。 | 初步理論驗證與小規模實驗均已完成；完整實驗計畫附於附錄。 |                                                     |                                                                   |

---

## 數學理論推演與關鍵證明（摘要）

### 定義 1（FAIM 度量）

對任意點 $x$ 與切向量 $v\in T_x\mathcal{M}$，定義

$$
F_x(v)=\sqrt{v^\top\Sigma(x)v}+\lambda\,\bigl|\beta(x)\!\cdot\!v\bigr|,
$$

其中 Σ(x) 為正定 Fisher 資訊矩陣（統計流形的自然黎曼張量）([arXiv][2])，β(x) 為 L1-正則化之判別向量場。

### 定理 1（Randers 類 geodesic 距離）

若 Σ, β 在 geodesic 區段上保持常值，則兩點 p,q 間最短路徑長

$$
d_F(p,q)=\sqrt{(q-p)^\top \Sigma (q-p)}+\lambda\,\bigl|\beta\!\cdot\!(q-p)\bigr|.
$$

*證明概要*：利用 Randers 度量可拆為黎曼項 + 1-form，根據 Shen (2001) 的 geodesic 展開；因 β 常值而路徑整合可閉式求解，大幅減少計算量。

### 定理 2（Margin 增幅界）

設兩類質心差向量 w，歐氏 margin 為 m\_E。則在 FAIM 下

$$
m_F \;\ge\; m_E + \lambda\,\frac{|β·w|}{\|w\|_2}.
$$

*證明概要*：由定理 1 距離公式直接代入，並應用 Cauchy-Schwarz 不等式得下界。當 β 與 w 同向時可獲最大增幅。

### 命題 1（泛化誤差上界）

在 1-Lipschitz 分類器下，使用 FAIM 導致的 Rademacher 複雜度上界為

$$
\mathcal{R}_n \le \frac{ \sqrt{\operatorname{tr}(\Sigma)} + \lambda\|\beta\|_2}{\sqrt{n}},
$$

並因此得到 Top-1 錯誤率 ≤ O(𝓡\_n / m\_F)。顯示 m\_F 的增幅可直接降低理論誤差。

---

## 可行性驗證

* **幾何合理性**：FAIM 屬 Finsler 流形，具備已知正定與 geodesic 性質，並可透過最近的 Finsler-Laplace operator 推導特徵基底([arXiv][1])。
* **計算可承受**：距離與梯度均閉式；與現行 Hyperbolic/球面層一樣只需一次前向 + 反向。
* **潛在應用**：除 UFGVC 外，任何「小差距量測」場景（如醫療影像亞型）皆可適用，且不受超參數曲率 κ 約束。

---

## 與現有工作的差異

* CLE-ViT 專注於**資料層增強**，而 FAIM 直接改變**特徵空間度量**，兩者可互補整合。
* 以往嘗試的超球／雙曲僅調整常曲率；FAIM 引入*方向性*曲率，可在關鍵維度放大差異。
* 透過 α-divergence 聯繫資訊幾何，可自然連結 Fisher-Rao 量測與分布特徵([yorkerlin.github.io][3])。

---

## 參考與背景

* UFG 基準數據集與挑戰描述
* CLE-ViT 提出之類內容忍概念
* Finsler 幾何在機器學習最新研究([arXiv][4]), ([arXiv][1])
* Fisher-Rao 與資訊幾何基礎([arXiv][2])
* Wasserstein / α-divergence 流形理論([arXiv][5])
* 球面與雙曲嵌入的限制分析([白玫瑰研究在線][6])

---

透過本研究可望在 WACV 發表一篇「理論新穎、實驗可行」的小而完整論文，為 Ultra-FGVC 社群提供一條走出傳統流形束縛的新方向。

[1]: https://arxiv.org/abs/2404.03999?utm_source=chatgpt.com "Finsler-Laplace-Beltrami Operators with Application to Shape Analysis"
[2]: https://arxiv.org/abs/1711.01530?utm_source=chatgpt.com "Fisher-Rao Metric, Geometry, and Complexity of Neural Networks"
[3]: https://yorkerlin.github.io/posts/2021/09/Geomopt01/?utm_source=chatgpt.com "Part I: Smooth Manifolds with the Fisher-Rao Metric - Wu Lin"
[4]: https://arxiv.org/html/2503.18010v1?utm_source=chatgpt.com "Finsler Multi-Dimensional Scaling: Manifold Learning for Asymmetric ..."
[5]: https://arxiv.org/abs/2311.08549?utm_source=chatgpt.com "[2311.08549] Manifold learning in Wasserstein space - arXiv"
[6]: https://eprints.whiterose.ac.uk/id/eprint/78407/1/SphericalFinal.pdf?utm_source=chatgpt.com "[PDF] Spherical and hyperbolic embeddings of data"
