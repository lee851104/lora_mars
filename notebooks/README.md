# notebooks/

| 檔案 | 用途 |
|---|---|
| `colab_train.ipynb` | **在 Colab 上跑完整流程**。它只負責呼叫 `src/`，不含任何邏輯 |
| `01_eda.ipynb` | 資料探索，唯讀 |

`colab_train.ipynb` 每一步都是 `!python -m src.xxx`，是獨立子行程——失敗時顯存會隨
行程結束自動釋放，不會累積成 OOM。**notebook 裡不放邏輯**，理由如下。

EDA 專用。**不要**把訓練、切分或評估邏輯寫在這裡。

理由：notebook 的執行順序不受約束，原始專案就是因為「訓練在前、切分在後」
這種只有在 notebook 裡才寫得出來的順序，造成測試集 100% 洩漏。

所有會影響結果的邏輯都放在 `src/`，notebook 只負責 import 與畫圖：

```python
from src.config import load_config
from src.data.build import load_processed
from src.data.split import load_manifest

cfg = load_config()
records = load_processed(cfg)
manifest = load_manifest(cfg)
```

圖表一律存到 `reports/figures/`，用 `src.data.validate.write_figures()`。
