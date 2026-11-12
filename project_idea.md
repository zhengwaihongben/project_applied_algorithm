# Betting on Traffic Jams in the Texax power grid

## Context

Profit from the price difference due to congestion, by trading Point-to-point(PTP) Obligation.

Day-Ahead Market gives a reference price. We predict the price too, and it a difference between our predicted price and the reference price appears, then trade.

## PTP

| Parameter | Value |
| --- | --- |
| Source |`HB_WEST` |
| Sink | `HB_NORTH` |
| Hour |  |
| Quantity | in `MW`|
| Maximum Price ||


# Stages

## Data

### Getting Data

From ERCOT
- Day-ahead PTP Obligation Price
- Real-time PTP Settlement Price
- Prediction on electricity production: wind-power level of West, temperature of North

### Data Processing

NaN, time ...

## Strategies

### Signal Generation

**Target**

**Extracting Features**

- Historical Congestion
- West Wind Forecast
- North Load Forecast
- Line Outages

**Model by Machine Learning**

### Entry/Exit Rules

Set the entry/exit rules

### Position Sizing

# Texas ERCOT Power Trading - Final Project 工作流程清單

## 專案概述
- **目標**：實作基於 PTP (Point-to-Point) Obligation 的擁擠價差交易策略，並產出回測結果。
- **核心邏輯**：預測明日的 Real-time 擁擠價差，並與 Day-ahead 拍賣價差進行比對。若 Edge > $5/MWh 則進場。
- **評分重點**：策略創意與邏輯 (10%)、風險控制 (10%)、簡報內容與學習經驗 (10%)、個人表達與 QA (10%)。**PnL 絕對不是唯一指標。**

---

### 【第一階段：資料獲取與清理】Day 1 - Day 3
- [ ] **任務 1**：註冊 ERCOT 官網，下載歷史數據（涵蓋 2023-2024 全年）。
  - 數據包含：Day-ahead PTP 價格、Real-time PTP 結算價、West 風力預測、North 氣溫/負載預測。
- [ ] **任務 2**：使用 Python (Pandas) 清理資料。
  - 處理缺失值、時間對齊（對齊至每小時 Hourly）。
- [ ] **任務 3**：輸出乾淨的 CSV 或 DataFrame 供後續使用。
- **分工建議**：2 人負責 Data (清理與整理)，2 人負責 Research (構思模型與邏輯)。

### 【第二階段：策略實作與回測引擎】Day 4 - Day 7
*(本階段是專案核心，細分為以下 5 個策略模組)*

- [ ] **模組 1：特徵工程與預測模型 (Signal)**
  - 定義 Target：預測明日的 Real-time Gap。
  - 選取 Features：昨日 Gap (均值回歸)、風力預測、氣溫預測、線路維修排程。
  - 建立預測模型（線性回歸或加權平均），輸出預測價差。
- [ ] **模組 2：進出場邏輯 (Entry/Exit)**
  - 計算 Edge = `預測 Real-time Gap - Day-ahead 拍賣 Gap`。
  - 進場規則：Edge > $5/MWh 買入 (West→North)；Edge < -$5/MWh 反向；其餘不交易。
  - 出場規則：持有至隔日現金結算 (Settlement)。
- [ ] **模組 3：部位規模與資金管理 (Position Sizing)**
  - 實作 Half-Kelly 準則 (0.5 * Kelly Formula)。
  - 設定硬性上限：單小時 10 MW、單日總量 40 MW。
  - 根據 Edge 大小動態調整下單量。
- [ ] **模組 4：風險控制機制 (Risk Management)**
  - 寫入 Bid Cap：最高願付價格 $25/MWh，超過則不送單。
  - 寫入 Daily Stop：昨日虧損 > 2% 則部位減半；> 5% 則禁止今日下單。
  - 寫入 Strategy Stop：總回撤 > 15% 則停止交易。
  - 寫入 Weather Filter：串接模擬極端天氣指標，觸發時減半或平倉。
- [ ] **模組 5：回測引擎與交易明細 (Backtesting)**
  - 計算每日現金流：`(Real-time 實際價差 - Day-ahead 支付價差) * 部位 MW`。
  - 產出 Trade Log (日期、方向、MW、支付價差、結算價差、盈虧)。
  - 注意避免未來函數 (Look-ahead bias)，可加入微量交易成本 (如 $0.5/MWh)。

- **分工建議**：1 人寫 Signal、1 人寫 Sizing & Risk、1 人寫 Backtest 引擎、1 人負責整合與 Debug。

### 【第三階段：結果分析與圖表製作】Day 8 - Day 9
- [ ] **任務 1**：繪製 PnL 曲線圖、最大回撤 (Drawdown) 圖。
- [ ] **任務 2**：計算績效指標：勝率 (Win rate)、平均獲利/虧損比、夏普比率 (Sharpe Ratio)。
- [ ] **任務 3**：挑選 2-3 筆代表性交易 (Case Study)，例如成功預測的事件，或觸發風控躲過大虧的案例。
- [ ] **任務 4**：總結「我們學到了什麼」(What is learned) - 這是評分重點。
- **分工建議**：全員參與，1 人畫圖，其他人協助撰寫簡報文字。

### 【第四階段：簡報製作與 QA 準備】Day 10 - Day 12
- [ ] **任務 1**：製作 PPT (15 分鐘內容，每人講 3-4 分鐘)。
  - 架構建議：市場背景 → 資料特徵與訊號 → 部位與風控 → 回測結果與學習心得。
- [ ] **任務 2**：撰寫個人講稿，並嚴格計時排練。
- [ ] **任務 3**：模擬 QA (5 分鐘)，互相提問尖銳問題。
  - 常見 QA 預測：如何避免 Overfitting？為什麼用 Half-Kelly？極端天氣下的滑價怎麼處理？

### 【第五階段：最終檢查與提交】Day 13 - Day 14
- [ ] **任務 1**：完整排練至少 2 次。
- [ ] **任務 2**：整理程式碼（必須加註解，建議用 Jupyter Notebook 圖文並茂）。
- [ ] **任務 3**：報告結束後，將 **簡報 (PPT)** 與 **程式碼 (Codes)** Email 給教授。
- [ ] **任務 4 (加分項)**：在台下聽其他組報告時，準備 1-2 個有深度的問題並舉手發問 (個人加分)。

---

## 教授評分重點與避坑指南
1. **PnL 不是唯一**：即使回測虧錢，只要在簡報中說明「我們透過風控（如 Daily Stop）成功限制了虧損，並學到了尾部風險的重要性」，就能拿高分。
2. **邏輯一致性**：PPT 上寫的風控規則，程式碼裡一定要有相對應的實作。
3. **展現演算法精神**：簡報時可以放幾段程式碼截圖 (Code Snippet)，證明你們不是手動交易，而是用演算法回測。
4. **誠實與反思**：教授看重「學習經驗」，如果策略失效，分析失效原因比偽造完美的 PnL 更有價值。