# notebooks/

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
