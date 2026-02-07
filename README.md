## 💰 Business Context: Cost Matrix Analysis

在半導體製造中，不同類型的預測錯誤會帶來不同的成本衝擊。本專案允許使用者透過 App 動態調整判定閾值（Threshold），以符合當下的商業策略。

### 成本矩陣 (Cost Matrix)

| 實際狀況 \ 預測結果 | 預測 Pass (0) | 預測 Fail (1) |
| :--- | :--- | :--- |
| **實際 Pass (0)** | ✅ **True Negative**<br>正常出貨<br>(Cost: $0) | ⚠️ **False Positive (誤殺)**<br>浪費重測成本/報廢良品<br>(Cost: Low ~ Medium) |
| **實際 Fail (1)** | ❌ **False Negative (漏檢)**<br>客戶退貨、賠償、信譽受損<br>(Cost: **Very High**) | ✅ **True Positive**<br>成功攔截瑕疵品<br>(Cost: Saved!) |

### 為什麼需要調整閾值 (Threshold)？
- **預設 (0.5)**：平衡準確率與召回率。
- **調低 (e.g., 0.3)**：**嚴格模式**。
  - 目的：寧可錯殺，不可放過。
  - 適用情境：車用電子、航太晶片等高可靠度要求產品。
  - 結果：Recall 上升（抓出更多瑕疵），但 False Positive 也會增加（誤殺良品）。
- **調高 (e.g., 0.7)**：**寬鬆模式**。
  - 目的：降低報廢率，節省成本。
  - 適用情境：消費性電子（如低階玩具晶片）。