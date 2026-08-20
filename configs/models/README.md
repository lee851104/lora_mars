# 模型預設

用 `MODEL=` 選：

```bash
make train MODEL=qwen2_vl_2b
make eval  MODEL=qwen2_vl_2b
make serve MODEL=qwen2_vl_2b
```

**同一次實驗的 train / eval / serve 要用同一個 `MODEL`**，否則 adapter 對不上基礎模型會直接載入失敗。

| 預設 | 授權 | 4-bit 大小 | T4 適合度 |
|---|---|---|---|
| `qwen2_5_vl_7b`（預設） | Apache 2.0 | 6.43 GiB | 好 |
| `qwen2_vl_2b` | Apache 2.0 | 2.29 GiB | 很好，上課首選 |
| `llama3_2_11b_vision` | Llama 3.2 Community | 7.37 GiB | 可以，但有授權限制 |

沒評估進來的：`Qwen2.5-VL-3B` 是 Qwen 自家研究授權（非商用），`Pixtral-12B` 雖然是
Apache 2.0 但 4-bit 就 8.58 GiB，在 T4 上比 7B 更緊。
