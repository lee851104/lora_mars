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
suggested_hardware: t4-small
short_description: Llama-3.2-11B-Vision + LoRA，天文影像描述，附微調前後對照
---

# 🔭 天文影像描述 · LoRA Demo

用 LoRA 在 250 張天文照片上微調 **Llama 3.2 11B Vision**，上傳圖片就會得到一段英文描述。
介面可以同時顯示**微調前**與**微調後**的輸出——兩者是同一個模型，只差有沒有掛上 LoRA adapter，
所以差異就是微調的效果。

## 這個 Space 需要的設定

| 項目 | 值 |
|---|---|
| Hardware | **T4 small 或更好**（4-bit 量化後約需 8 GB VRAM） |
| Space variable | `LORA_REPO_ID` = `<user>/<adapter-repo>` |

**免費的 CPU basic 硬體跑不動這個模型**，一定要換 GPU。ZeroGPU 可行但需要 PRO 訂閱，
而且每次呼叫都要重新把 8 GB 權重搬上 GPU，第一次會很慢。

`LORA_REPO_ID` 沒設的話，介面會明確標示「現在跑的是原廠模型」，不會假裝是微調結果。

## 限制

這是教學／作品集用途的 demo，**不能當科學判讀依據**。訓練資料只有 250 筆、預設只訓練
30 步，模型學到的主要是輸出風格而不是新的天文知識。完整的限制、已知偏誤與不適用情境
寫在專案的 `MODEL_CARD.md`。

## 原始碼

<https://github.com/lee851104/lora_mars>
