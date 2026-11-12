import os
import pandas as pd
import ercot as er
from pathlib import Path

# ==========================================
# 1. 設定你的 API 憑證
# ==========================================
# 從環境變數讀取，切勿把真實帳密寫進程式碼或 commit
USERNAME = os.environ["ERCOT_USERNAME"]
PASSWORD = os.environ["ERCOT_PASSWORD"]
SUBSCRIPTION_KEY = os.environ["ERCOT_SUBSCRIPTION_KEY"]

# 初始化連線 (只需執行一次，憑證會存到本機)
er.configure(USERNAME, PASSWORD, SUBSCRIPTION_KEY)

# ==========================================
# 2. 定義要下載的參數與路徑
# ==========================================
# 建議先抓短區間測試，格式：YYYY-MM-DD
START_DATE = "2026-01-01"
END_DATE = "2026-01-31" 

# 你要的結算點 (Hubs)
POINTS = ["HB_WEST", "HB_NORTH"]

# 根據你的資料夾結構設定輸出路徑
BASE_DIR = Path(__file__).resolve().parent.parent  # 回到 project 根目錄
DAM_DIR = BASE_DIR / "data" / "raw" / "DAM"
RTM_DIR = BASE_DIR / "data" / "raw" / "RTM"

# 確保資料夾存在
DAM_DIR.mkdir(parents=True, exist_ok=True)
RTM_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. 批量下載函數
# ==========================================
def download_and_save():
    print(f"開始下載資料：{START_DATE} 至 {END_DATE}")
    
    # --- 下載 DAM (Day-Ahead Market) ---
    print("正在下載 DAM 資料...")
    for point in POINTS:
        try:
            # er.dam() 預設回傳寬表 (Wide Format, 每小時一欄)
            df_dam = er.dam(point, START_DATE, END_DATE)
            save_path = DAM_DIR / f"DAM_{point}_{START_DATE}_{END_DATE}.csv"
            df_dam.to_csv(save_path, index=False)
            print(f" DAM 已儲存: {save_path.name}")
        except Exception as e:
            print(f" DAM 下載失敗 ({point}): {e}")

    # --- 下載 RTM (Real-Time Market) ---
    print("正在下載 RTM 資料...")
    for point in POINTS:
        try:
            # er.rtm() 預設回傳長表 (Long Format, 每 15 分鐘一筆)
            # 可以視需要加上 format="wide" 來取得寬表，但長表通常更適合後續聚合
            df_rtm = er.rtm(point, START_DATE, END_DATE, format="long")
            save_path = RTM_DIR / f"RTM_{point}_{START_DATE}_{END_DATE}.csv"
            df_rtm.to_csv(save_path, index=False)
            print(f" RTM 已儲存: {save_path.name}")
        except Exception as e:
            print(f" RTM 下載失敗 ({point}): {e}")

if __name__ == "__main__":
    download_and_save()