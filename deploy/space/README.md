---
title: 天文影像描述 LoRA
emoji: 🔭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: other
python_version: 3.10
short_description: Gemma 3 4B + LoRA，天文影像描述，附微調前後對照
---

# 🔭 天文影像描述 · LoRA Demo

用 LoRA 在 250 張天文照片上微調 **Gemma 3 4B IT**，上傳圖片就會得到一段英文描述。
介面可以同時顯示**微調前**與**微調後**的輸出——兩者是同一個模型，只差有沒有掛上 LoRA adapter，
所以差異就是微調的效果。

## 這個 Space 需要的設定

| 項目 | 值 |
|---|---|
| Hardware | **ZeroGPU**（免費 demo）或 T4 small 以上（常駐、付費） |
| Space variable | `LORA_REPO_ID` = `<user>/<adapter-repo>` |

**免費的 CPU basic 硬體跑不動這個模型**。個人免費帳號可用 **ZeroGPU**：帳號需通過
email 驗證且建立超過 30 天，最多可建立兩個 ZeroGPU Space；每位免費使用者每日有 5 分鐘
GPU 額度，可能排隊。第一次呼叫要下載並載入約 4.3 GB 基礎模型，會比較慢。

ZeroGPU 適合課程展示與作品集，不適合多人長時間使用。需要穩定常駐時，改選 T4 small 或更好，
但 GPU 時間會計費。

`LORA_REPO_ID` 沒設的話，介面會明確標示「現在跑的是原廠模型」，不會假裝是微調結果。

## 限制

這是教學／作品集用途的 demo，**不能當科學判讀依據**。訓練資料只有 250 筆、預設只訓練
30 步，模型學到的主要是輸出風格而不是新的天文知識。完整的限制、已知偏誤與不適用情境
寫在專案的 `MODEL_CARD.md`。

## 授權

基礎模型 **Gemma 3 4B IT** 使用受 [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms)
與 [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) 約束；
再散布時必須一併傳遞這些使用限制。

> 換模型就換授權義務：`qwen2_5_vl_7b` 是 Apache 2.0（無額外條款、無地域限制）；
> `llama3_2_11b_vision` 必須顯示 **"Built with Llama"**，而且 Llama 3.2 的視覺功能
> 在歐盟境內的自然人或法人不得使用。介面的「說明」分頁會依設定自動顯示對應的授權文字。

## 原始碼

<https://github.com/lee851104/lora_mars>
