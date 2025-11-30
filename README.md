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
- `compute_token_heatmap.py` : 특정 토큰(기본 `jacket`)의 레이어/헤드별 어텐션을 집계해 24×32 서브패치 히트맵과 원본 이미지 오버레이 PNG 생성

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

## 토큰별 집계 히트맵/오버레이 생성 (compute_token_heatmap.py)
```bash
python compute_token_heatmap.py \
  --token jacket \
  --decoded-tokens decoded_tokens.npy \
  --attn-dir attentions_5 \
  --layers 0-29 \
  --topk 10 \
  --output token_heatmap.png \
  --overlay token_heatmap_overlay.png \
  --overlay-alpha 0.5
```
- 처리: 각 레이어/헤드의 토큰 어텐션을 상위 `topk`만 사용해 재정규화하고, 3×4 패치 × 8×8 서브패치 = 24×32 그리드로 집계
  - 가중치: `attn_sum × (1 - entropy_norm)` (sum이 클수록, 엔트로피가 낮을수록 영향도 ↑)
  - 엔트로피 정규화: 레이어/헤드별 분포 엔트로피를 `log(N)`으로 나누어 0~1 구간으로 만든 뒤 (1 - 값)으로 뒤집어 사용
- 결과: 히트맵(`token_heatmap.png`)과 원본 이미지에 최근접 보간으로 격자 구조를 유지한 오버레이(`token_heatmap_overlay.png`)
