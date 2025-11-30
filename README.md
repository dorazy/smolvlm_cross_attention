# SmolVLM Cross Attention Viewer

SmolVLM(Instruct-256M)으로 이미지별 어텐션/토큰을 추출하고, 웹 UI에서 토큰별 어텐션 히트맵을 시각화하는 프로젝트입니다. 단일/일괄 추출 스크립트, Flask 뷰어, 간단 벤치마크 스크립트를 포함합니다.

## Flask 뷰어 빠르게 띄우기 (app.py)
- 필요 파일: `decoded_tokens.npy`, `attentions_5/attention_fp16_rounded_layer_0..29.npz`, `image.png`(또는 `static/image.png`로 복사됨)
- 실행:
  ```bash
  pip install -r requirements.txt   # 필요 시
  python app.py                     # http://localhost:5001
  ```
- 동작 개요: `/api/initialize`에서 `decoded_tokens.npy`와 첫 번째 레이어를 읽어 토큰 목록/헤드 수를 계산하고, `/api/generate_attention_map`이 요청 시 지정한 레이어의 `.npz`를 불러와 히트맵을 생성합니다. 원본 이미지는 루트의 `image.png`를 `static/image.png`로 복사해 사용합니다.

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
