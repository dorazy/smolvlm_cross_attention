# SmolVLM Cross Attention Viewer

SmolVLM(Instruct-256M)으로 이미지별 어텐션/토큰을 추출하고, 웹 UI에서 토큰별 어텐션 히트맵을 시각화하는 프로젝트입니다. 단일/일괄 추출 스크립트, Flask 뷰어, 간단 벤치마크 스크립트를 포함합니다.

## 주요 파일
- `extract_smolvlm_attn.py` : 단일 이미지 어텐션/토큰 추출
- `extract_smolvlm_batch.py` : 여러 이미지 일괄 추출
- `app.py` : Flask 웹앱 (토큰 클릭 → 어텐션 히트맵)
- `benchmark.py` : POPE 포맷 질문 파일로 간단 벤치마크

## 환경 준비
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade torch transformers --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 어텐션/토큰 추출 (CPU 예시)
### 단일 이미지
```bash
python extract_smolvlm_attn.py \
  --image image.png \
  --output-dir . \
  --prompt "Can you describe this image?" \
  --model-id HuggingFaceTB/SmolVLM-Instruct-256M \
  --device cpu
```
- 생성물: `decoded_tokens.npy`, `attention_array.npy`, `attentions_5/attention_fp16_rounded_layer_*.npz`

### 일괄 추출
```bash
python extract_smolvlm_batch.py \
  --image-dir ../COCO/val2014 \
  --output-root smol_outputs \
  --prompt "Can you describe this image?" \
  --model-id HuggingFaceTB/SmolVLM-Instruct-256M \
  --device cpu \
  --limit 0   # 0이면 전체
```
- 이미지별 폴더(`smol_outputs/<이미지스텀>/`)에 토큰/어텐션 저장.

## 웹 뷰어 실행
```bash
source .venv/bin/activate
python app.py
# 브라우저에서 http://localhost:5001
```
- 토큰을 클릭하면 선택한 레이어/헤드의 어텐션 히트맵이 원본 이미지에 오버레이됩니다.
- 화면 표시용으로 BPE 마커(Ġ, Ċ, ▁)는 치환되지만, 백엔드 인덱싱은 원본 토큰을 사용합니다.

## 벤치마크 (POPE 샘플)
```bash
python benchmark.py \
  --benchmark-file ../POPE/output/coco/coco_pope_popular_100.json \
  --image-root ../COCO/val2014 \
  --output-dir benchmark_output \
  --layer 20 --head 5 --threshold 0 --limit 10
```
- 결과: `benchmark_output/summary.jsonl`, `predictions.jsonl`, `benchmark_output/images/`(히트맵/오버레이)

## 주의사항
- 현재 `attention_array.npy`는 `(30, 9, 1148, 1148)`이며, `decoded_tokens.npy` 및 `attentions_5/*.npz`와 세트로 사용해야 합니다.
- `attention_array.npy`, `attentions_5/*.npz`는 수백 MB 단위의 대용량 파일입니다. 이동/커밋 시 용량을 고려하세요.
- GPU가 있으면 `--device cuda`로 속도를 높일 수 있고, VRAM 부족 시 CPU로 실행하세요.
