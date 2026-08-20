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
| `gemma3_4b` | Gemma Terms of Use | 4.25 GiB | 好，但 fp16 數值最敏感 |
| `qwen2_vl_2b` | Apache 2.0 | 2.29 GiB | 很好，上課首選 |
| `llama3_2_11b_vision` | Llama 3.2 Community | 7.37 GiB | 可以，但有授權限制 |

### Gemma 3 要注意的一件事

Gemma 3 用 bf16 訓練，activation 會超過 float16 的上限（65504），而 T4 沒有 bf16
硬體。unsloth 對此有專門處理（activation 走 bf16/fp32、只有 matmul 降到 fp16、
layernorm 升到 fp32），所以在 T4 上可以跑，但這是四個預設裡最容易出現 `nan` loss 的
一個。看到 loss 變 nan：先確認 unsloth 是最新版，再把 `train.learning_rate` 往下調
（預設檔已經先調到 1e-4）。

沒評估進來的：`Qwen2.5-VL-3B` 是 Qwen 自家研究授權（非商用），`Pixtral-12B` 雖然是
Apache 2.0 但 4-bit 就 8.58 GiB，在 T4 上比 7B 更緊。
