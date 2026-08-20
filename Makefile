# Llama-3.2-11B-Vision LoRA — 天文影像描述
#
# 流程順序是刻意的：features（含切分）一定在 train 之前。
# 訓練只吃 data/splits/train.json，evaluate 只吃 data/splits/test.json，
# 所以「先訓練後切分」造成的測試集洩漏在結構上不可能發生。
#
# Windows 沒有 make 的話，直接照著每個 target 底下的 uv 指令跑即可。

PY      := uv run python
CONFIG  ?= configs/config.yaml
OVERRIDE ?=

# MODEL 選 configs/models/<name>.yaml 這個預設檔。可用的看 configs/models/README.md。
#   make train MODEL=qwen2_vl_2b
# 同一次實驗的 train / eval / serve 要用同一個 MODEL，否則 adapter 對不上基礎模型。
MODEL   ?=
MODEL_ARG := $(if $(MODEL),--override-file configs/models/$(MODEL).yaml,)
ARGS    := --config $(CONFIG) $(MODEL_ARG) $(OVERRIDE)

.PHONY: help setup setup-gpu data features train eval eval-base score serve weights upload space test lint fmt clean all
.DEFAULT_GOAL := help

help:
	@echo "make setup     安裝依賴（CPU 基礎 + dev）"
	@echo "make setup-gpu 安裝 GPU 訓練堆疊（Colab / 有 CUDA 的機器）"
	@echo "make data      下載原始資料集到 data/raw/"
	@echo "make features  清洗 + 驗證 + 切分 -> data/processed/ 與 data/splits/"
	@echo "make train     LoRA 微調（只讀 train 切分）"
	@echo "make eval      在 test 切分上產生預測並評分（CLIPScore + LLM-as-judge）"
	@echo "make eval-base 同上但不掛 LoRA，產生成果表的 Base 對照欄"
	@echo "make score     只重算指標，不重新產生預測"
	@echo "make weights   從 Hugging Face 下載 LoRA 權重到 models/lora/"
	@echo "make serve     啟動 Gradio 介面"
	@echo "make upload    把 models/lora/ 推到 Hugging Face"
	@echo "make space     組出 build/space/，準備推到 Hugging Face Space"
	@echo "make test      跑測試（含 test_no_leakage.py）"
	@echo "make all       data -> features -> train -> eval"
	@echo ""
	@echo "MODEL=<name>   換基礎模型，見 configs/models/README.md"
	@echo "               預設 qwen2_5_vl_7b（Apache 2.0）；上課用 qwen2_vl_2b"

setup:
	uv sync --extra dev --extra clip --extra judge --extra serve

setup-gpu:
	uv sync --extra dev --extra clip --extra judge --extra serve --extra gpu

data:
	$(PY) -m src.data.download $(ARGS)

features:
	$(PY) -m src.data.build $(ARGS)
	$(PY) -m src.data.split $(ARGS)

train: features
	$(PY) -m src.models.train $(ARGS)

eval:
	$(PY) -m src.models.evaluate $(ARGS)

# 成果表的 Base 對照欄：只跑基礎模型，輸出檔名自動加 _base
eval-base:
	$(PY) -m src.models.evaluate $(ARGS) eval.use_adapter=false

# 只重算指標，不重新產生預測（讀 reports/predictions_<split>.jsonl）
score:
	$(PY) -m src.models.score $(ARGS)

weights:
	$(PY) -m src.models.download $(ARGS)

serve:
	$(PY) -m src.serving.app $(ARGS)

# Colab 上想要對外連結就加 OVERRIDE="serving.share=true"
upload:
	$(PY) -m src.models.upload $(ARGS)

space:
	$(PY) -m src.serving.build_space $(ARGS)

test:
	uv run pytest

lint:
	uv run ruff check src tests

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

clean:
	rm -rf data/processed data/splits outputs reports/*.json reports/*.jsonl
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

all: data features train eval
