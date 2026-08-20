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

.PHONY: help setup setup-gpu data features train eval serve weights upload space test lint fmt clean all
.DEFAULT_GOAL := help

help:
	@echo "make setup     安裝依賴（CPU 基礎 + dev）"
	@echo "make setup-gpu 安裝 GPU 訓練堆疊（Colab / 有 CUDA 的機器）"
	@echo "make data      下載原始資料集到 data/raw/"
	@echo "make features  清洗 + 驗證 + 切分 -> data/processed/ 與 data/splits/"
	@echo "make train     LoRA 微調（只讀 train 切分）"
	@echo "make eval      在 test 切分上算 BLEU/ROUGE（含 bootstrap 信賴區間）"
	@echo "make weights   從 Hugging Face 下載 LoRA 權重到 models/lora/"
	@echo "make serve     啟動 Gradio 介面"
	@echo "make upload    把 models/lora/ 推到 Hugging Face"
	@echo "make space     組出 build/space/，準備推到 Hugging Face Space"
	@echo "make test      跑測試（含 test_no_leakage.py）"
	@echo "make all       data -> features -> train -> eval"

setup:
	uv sync --extra dev --extra eval --extra serve

setup-gpu:
	uv sync --extra dev --extra eval --extra serve --extra gpu

data:
	$(PY) -m src.data.download --config $(CONFIG) $(OVERRIDE)

features:
	$(PY) -m src.data.build --config $(CONFIG) $(OVERRIDE)
	$(PY) -m src.data.split --config $(CONFIG) $(OVERRIDE)

train: features
	$(PY) -m src.models.train --config $(CONFIG) $(OVERRIDE)

eval:
	$(PY) -m src.models.evaluate --config $(CONFIG) $(OVERRIDE)

weights:
	$(PY) -m src.models.download --config $(CONFIG) $(OVERRIDE)

serve:
	$(PY) -m src.serving.app --config $(CONFIG) $(OVERRIDE)

# Colab 上想要對外連結就加 OVERRIDE="serving.share=true"
upload:
	$(PY) -m src.models.upload --config $(CONFIG) $(OVERRIDE)

space:
	$(PY) -m src.serving.build_space --config $(CONFIG) $(OVERRIDE)

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
