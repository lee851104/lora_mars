# Model Card — AstroVision LoRA

## 概要

| 項目 | 內容 |
|---|---|
| 基礎模型 | `unsloth/Llama-3.2-11B-Vision-Instruct`（4-bit bnb 量化，約 7.9 GB） |
| 微調方式 | LoRA（預設 r=16, alpha=16, dropout=0），視覺與語言層皆掛 adapter |
| 訓練資料 | `AIOmarRehan/space-multimodal-dataset`，250 組影像–caption |
| 訓練規模 | 預設 `max_steps=30`、有效 batch 8 → 約 240 個樣本次，**連一個 epoch 都沒跑完** |
| 任務 | 單張天文影像 → 一段英文描述 |
| 硬體 | 單張 Tesla T4（16 GB）可訓練與推論 |
| 授權 | 依循 Llama 3.2 Community License |

## 預期用途

- 展示 vision-language 模型的 LoRA 微調流程（教學／作品集）。
- 對星體、行星表面、探測器影像產生**風格接近訓練資料**的英文描述。
- 作為「風格遷移是否成功」的對照實驗基準。

## 不適用情境

**不要**把這個模型用在以下任何一種場合：

- **科學判讀或測量**。它不會估計距離、光度、紅移、成分，輸出的任何數值都是文字模仿，不是量測。
- **天體識別／分類的權威依據**。它可能把仙女座說成銀河系，把火星地表說成月球。
- **教育教材直接發佈**。輸出未經事實查核，錯誤描述會直接誤導學習者。
- **非天文影像**。訓練資料只有五類天文場景，餵進人像、街景、醫療影像會得到自信但荒謬的天文式描述。
- **英文以外的語言**。訓練資料全英文，其他語言輸出未經評估。
- **任何需要校準信心度的決策流程**。模型不會說「我不確定」，語氣一律肯定。

## 已知限制

### 資料量極小
250 筆，去重後更少。切出 held-out 之後只有二十幾筆，**任何單一指標的點估計都不可靠**。
評估一律附 bootstrap 95% 信賴區間，請看區間重疊與否，不要比較小數點後三位。

### 訓練不足
預設 `max_steps=30` 是為了讓 T4 在課堂時間內跑完，不是收斂設定。這個模型主要學到的是
**輸出風格**（句長、用語、句型），不是新的天文知識。

### 類別嚴重不平衡
啟發式標籤在原始 250 筆上的分佈：

| 標籤 | 筆數 |
|---|---|
| Earth | 77 |
| Mars | 54 |
| Mars Rover | 46 |
| Milky Way | 45 |
| Hubble | 28 |

Hubble 類只有 Earth 的三分之一，模型在該類上的表現預期較差，而 held-out 裡該類可能只有
兩三筆——**單看整體平均會蓋掉這件事**。切分有做分層抽樣（`split.stratify_by: label`），
但樣本數本身無法靠抽樣補救。

### 標籤本身就是啟發式
`heuristic_label()` 只是關鍵字比對，用於分層抽樣與資料分佈圖，**不是訓練目標，也不是
ground truth**。任何沒命中關鍵字的 caption 都會落到 `Unknown`。

### 清洗會造成資訊損失
`features.clean_mode` 預設 `conservative`（只正規化空白）。原專案用的 `aggressive` 模式會
把所有非英文字母字元刪掉，`M31`、`Apollo 11`、`2019` 這類天文專有名詞會**永久殘廢**，
模型也會學著輸出同樣殘缺的文字。保留該模式只為了能重現原專案結果，不建議使用。

### 指標本身的限制
BLEU 是 corpus-level n-gram 重疊指標，對「同義但換句話說」的正確描述會給低分；
ROUGE 偏向 recall，長輸出容易佔便宜。兩者都**量不到事實正確性**。
一個把星系名稱全講錯但句型一致的輸出，分數可以很高。

## 已知偏誤

- **地球中心偏誤**：訓練資料近三分之一是地球影像，模型傾向把不確定的行星表面描述成地球。
- **探測器偏誤**：資料集裡的火星影像大量伴隨 Curiosity/Perseverance，模型看到火星地表容易
  主動補上不存在的探測器。
- **語氣過度肯定**：訓練 caption 全部是斷言句，模型因此永遠不表達不確定性。
- **英文單一語域**：所有 caption 出自同一資料來源，句型高度同質，模型會把這種句型當成
  「正確答案的樣子」。

## 評估方式

- 切分：train / val / test，預設 80 / 10 / 10，分層依 `label`，seed 固定。
- **held-out 從未參與訓練**：切分先於訓練發生，切分歸屬寫進 `data/splits/manifest.json`，
  其 `split_hash` 會被 stamp 進 `models/lora/train_meta.json`；`tests/test_no_leakage.py`
  驗證三個切分互斥、聯集完整，且已訓練模型記錄的 hash 與現行 manifest 一致。
- 解碼：greedy（`do_sample=false`），確保可重現。
- 計分：只對**新生成的 token** 計分，prompt 在 decode 前就被切掉。
- 區間：BLEU 用 bootstrap 重抽整個語料，ROUGE 用 per-sample 分數重抽，各 1000 次取 2.5/97.5 百分位。

實際數字見 `reports/eval_test.json`。README 的成果表在 `make eval` 跑完前是空的——
**沒跑就不填數字**。

## 環境注意事項

- `transformers` 釘 `4.57.3`。`4.56.2` 搭上 `huggingface_hub` 1.x 時，載入 processor
  會去探測不存在的 `additional_chat_templates/` 目錄，hub 1.x 對此拋出
  `RemoteEntryNotFoundError` 而 4.56.2 沒有攔截，processor 就變成 `None`。
  4.57.x 的解法是把 `huggingface-hub` 上限鎖在 `<1.0`，等於從依賴層面消除這個組合
  （整條 4.x 線都不支援 hub 1.x；要用 hub 1.x 得跳到 transformers 5.2.0～5.5.0，
  但那是 major 改版，processor 與 `SFTConfig` 介面都有變動）。
  `4.57.0`／`4.57.4`／`4.57.5` 被 unsloth 的 metadata 排除，所以取 `4.57.3`。
- 在 T4 上，一次失敗的載入會留下約 7.9 GB 殘留顯存，直接重跑必定 OOM。
  `assert_gpu_headroom()` 會先擋下並提示重啟 runtime。

## 引用

```
基礎模型：Meta Llama 3.2 11B Vision Instruct（經 Unsloth 4-bit 量化）
資料集：AIOmarRehan/space-multimodal-dataset
流程參考：Unsloth_Llama_3.2_11B_Vision_Instruct_Astronomy
```
