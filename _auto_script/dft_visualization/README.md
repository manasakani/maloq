# Density and Hamiltonian visualizer

Quasar CPU에서 사용하던 `dft-monitor` 정적 UI와 `dft-dataset` FastAPI 계산
backend를 SC26에서 그대로 재사용한다. 새 renderer를 복제하지 않고, 이 디렉터리는
환경 준비·검증·MALOQ AO convention 변환만 담당한다.

## 표현 원칙

- Hamiltonian은 3D scalar field가 아니므로 AO heatmap, atom/shell block,
  diagonal/on-site energy, eigenspectrum, GT/pred/diff, parity/error 분포로 본다.
- Density는 AO density matrix `P`와 real-space field
  `rho(r) = sum(P_mn phi_m(r) phi_n(r))`를 함께 본다.
- sample JSON에 `density_matrix`가 있으면 직접 예측한 density를 사용한다.
  없으면 generalized eigensolve `H C = S C epsilon` 뒤 occupation으로 `P`를 만든다.
- 모든 dashboard 입력 행렬은 최종적으로 PySCF real-spherical AO order를 사용한다.
  adapter가 MALOQ e3nn/storage order를 변환한다.

## 구성

- UI: `/dataset/seongsu/shared-home/projects/dft-monitor`
- API/calculation: `/dataset/seongsu/shared-home/projects/dft-dataset/server.py`
- Quasar에서 가져온 sample: `dft-monitor/data/` (36개, 약 28 MB)
- runtime/cache: `/dataset/seongsu/shared-home/workspace/project/outputs/dft-visualization`
- web environment: `/dataset/seongsu/shared-home/conda/envs/proj-dft-visualization-sc26`

## 실행

최초 한 번:

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/dft_visualization/run_visualizer.sh prepare
```

검증:

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/dft_visualization/run_visualizer.sh validate
```

foreground server:

```bash
/dataset/seongsu/shared-home/workspace/project/_auto_script/dft_visualization/run_visualizer.sh serve
```

기본 bind는 `127.0.0.1:9100`이다. 로컬 PC에서 다음처럼 tunnel을 연 뒤
`http://127.0.0.1:9100/detail.html?id=qh9_000002`에 접속한다.

```bash
ssh -L 9100:127.0.0.1:9100 scp-gpu-1
```

다른 port는 `DFT_VIS_PORT=19100`, 외부 interface bind가 꼭 필요할 때만
`DFT_VIS_HOST=0.0.0.0`을 명시한다.

## MALOQ sample 등록

입력 NPZ에는 `atomic_numbers`, `positions`, `hamiltonian`이 필요하고,
`overlap`, `density_matrix`는 선택이다. overlap이 없으면 PySCF로 계산한다.
직접 density prediction을 보려면 reference H/S와 predicted `density_matrix`를 한
NPZ에 넣는다.

```bash
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
$PY /dataset/seongsu/shared-home/workspace/project/_auto_script/dft_visualization/export_maloq_visualization.py sample \
  /path/to/full_matrices.npz \
  --sample-id my_density_prediction \
  --matrix-convention maloq-e3nn
```

ASE DB의 loader 이전 storage matrix를 직접 넣는 경우
`--matrix-convention maloq-storage`를 사용한다. QH9 sample처럼 이미 PySCF
order이면 `pyscf`를 사용한다. 기존 sample ID는 `--force` 없이는 덮어쓰지 않는다.

## Hamiltonian prediction 비교

MALOQ e3nn-order prediction을 dashboard comparison용 PySCF NPZ로 변환한다.

```bash
PY=/dataset/seongsu/shared-home/conda/envs/proj-dft-baselines-maloq-sc26/bin/python
$PY /dataset/seongsu/shared-home/workspace/project/_auto_script/dft_visualization/export_maloq_visualization.py prediction \
  /path/to/prediction.npz \
  --reference /dataset/seongsu/shared-home/projects/dft-monitor/data/samples/qh9_000002.json \
  --output /dataset/seongsu/shared-home/workspace/project/outputs/dft-visualization/predictions/qh9_000002.npz \
  --matrix-convention maloq-e3nn \
  --model-name MALOQ
```

Dashboard의 `Comparison` 탭에서 위 server path를 입력하면 GT/pred/diff,
per-block MAE, eigenvalue parity와 DOS overlay를 볼 수 있다.

## 입력 contract

- positions: Angstrom
- H/Fock: Hartree
- overlap: dimensionless
- density matrix: dimensionless AO matrix
- basis: 현재 UI의 상세 AO label은 `def2-svp` 중심
- closed shell: `(nao, nao)`
- unrestricted: `(2, nao, nao)` with alpha/beta first axis

Density-only sample도 matrix/grid 계산은 가능하지만 현재 UI의 orbital/eigenspectrum
탭은 Hamiltonian을 요구하므로 H/S를 함께 제공하는 것을 기본 contract로 둔다.
