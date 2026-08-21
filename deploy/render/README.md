# Render + NVIDIA API 互動展示

這個部署不在 Render 載入 Gemma 3 4B，也不使用 LoRA。Render 只執行輕量的 Gradio
介面與 API 代理；圖片會由伺服器縮小後送到 NVIDIA 託管的
`meta/llama-3.2-11b-vision-instruct`。

## 部署

1. 先把本 repository 推到 GitHub。
2. 到 <https://dashboard.render.com/>，選 **New → Blueprint**。
3. 連接此 GitHub repository；Render 會讀取根目錄的 `render.yaml`。
4. 在建立畫面填入 `NVIDIA_API_KEY`，值貼 NVIDIA Build 取得的 API 金鑰。
5. 建立完成後，開啟 Render 提供的 `*.onrender.com` 網址。

金鑰只能放在 Render 的 Secret／Environment Variable，不能寫進程式、`.env`、GitHub，
也不能放在前端 JavaScript。

若要限制訪客消耗 NVIDIA API 額度，可在 Render 的 **Environment** 再新增：

```text
DEMO_ACCESS_CODE=<自己設定的密碼>
```

設好後頁面會自動顯示存取碼欄位。不設定就是公開使用。

## 本機確認

先複製 `.env.example` 成 `.env`，再填入：

```text
NVIDIA_API_KEY=nvapi-你的金鑰
```

`.env` 已被 `.gitignore` 排除，不能手動強制提交。接著執行：

```bash
pip install -r requirements-render.txt
python -m src.serving.nvidia_app
```

也可以不使用 `.env`，直接由 PowerShell 設 `$env:NVIDIA_API_KEY="<your-key>"`。
不要把真實金鑰貼進 issue、截圖或 commit。

## 限制

- Render Free Web Service 閒置後會休眠，第一次開啟可能需要等待喚醒。
- NVIDIA Build 託管 API 是開發／試用服務，可能遇到速率或額度限制，不應視為正式 SLA。
- 這不是 `lee851104/gemma3-4b-astronomy-lora` 的輸出。未來要展示微調成果，仍需可執行
  Gemma 3 4B + LoRA 的 GPU 服務，例如 Hugging Face ZeroGPU。
- Built with Llama；模型使用受 Llama 3.2 Community License 與 Acceptable Use Policy 約束。
