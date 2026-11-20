# Handwritten music: training the symbol detector

This folder trains the model that will let the converter read handwritten parts. The background, options and licenses are in [RESEARCH.md](RESEARCH.md).

**Plan:** train a general handwritten model on Kaggle's free GPU, then fine-tune it on your copyist's own pages.

## What's here

| File | Purpose |
|---|---|
| `classes.py` | The 47 symbol classes, the mapping from MUSCIMA++ classes, and the tile geometry (staff space = 22 px, tiles 1024×320). |
| `prepare_muscima.py` | Downloads MUSCIMA++ v2 and its images, cuts each staff into tiles, and splits by writer (train 1–40, val 41–45, test 46–50). |
| `synth_petaluma.py` | Synthetic handwritten-style staff strips in the Petaluma font, with exact boxes. |
| `prepare_copyist.py` | Your copyist's pages: tile them, pre-label with a model, then split the corrected labels by page. |
| `merge_coco.py` | Combines all sources into the `instances_*.json` files YOLOX reads. Copyist pages are repeated 5× in training. |
| `yolox_handwritten.py` | YOLOX-Tiny/Nano settings: no flips, almost no rotation, up to 400 labels per tile. |
| `export_model.py` | ONNX export (fp32 + fp16) plus `music_symbols.json` for the browser. |
| `per_class_report.py` | Per-class precision and recall on held-out pages. |
| `staves.py` | Staff finder and tiler for scans. |
| `viz.py` | Draws boxes on tiles so you can check labels by eye. |
| `kaggle_train.ipynb` | The whole pipeline, ready to run on Kaggle. |

## Stage 1: base model (no copyist pages needed)

1. Create a free account at kaggle.com and verify your phone. Phone verification is needed for internet access and GPUs.
2. Go to **Create → New Notebook → File → Import Notebook** and upload `kaggle_train.ipynb`.
3. In the side panel, set **Accelerator: GPU T4 ×1** and **Internet: On**.
4. Click **Save Version → Save & Run All (Commit)**. It runs in the background, so you can close the browser.
5. When it finishes, the **Output** tab holds `export/`:
   - `music_symbols.onnx` and `music_symbols.fp16.onnx`
   - `music_symbols.json`
   - `best_ckpt.pth` (keep this for stage 2)
   - `test_report.txt`, the per-class accuracy on writers the model never saw

Kaggle allows about 30 GPU hours a week. One run should fit comfortably, but the run time is an estimate until the first run.

## Stage 2: your copyist

1. Scan 20–50 pages at **300 dpi**, grayscale, flat and straight. More variety (different pieces) beats more pages of the same piece.
2. Tile and pre-label them on your computer. You need Python with `pip install pillow numpy onnxruntime`.
   ```
   python prepare_copyist.py tile --pages "scans/*.png" --out copyist_work --onnx music_symbols.onnx
   ```
3. Correct the labels in an annotation tool that imports and exports **COCO 1.0**, such as CVAT. Load `copyist_work/images` and `copyist_work/prelabels_coco.json`, fix wrong boxes and classes, add missed symbols, then export as COCO 1.0. This is the slow part: expect several minutes per page even with pre-labels.
4. Split the corrected pages:
   ```
   python prepare_copyist.py finalize --coco corrected.json --images copyist_work/images --out copyist_dataset
   ```
5. Upload `copyist_dataset/` as a Kaggle Dataset, and also upload stage 1's `best_ckpt.pth`. In the notebook, set `COPYIST_DIR`, `BASE_CKPT` and `EPOCHS = 40`, then run it again. `test_report.txt` now includes the copyist's held-out pages, which is the number that matters for you.

## Stage 3: use the model in the converter

Open the converter, click **Load model (.onnx)** under "Handwritten music" and pick `music_symbols.onnx`. The fp16 file also works and is half the size. The model is stored in your browser, so you only load it once.

"Use it" has three settings:
- **When the print rules fail** (default): files are read with the print rules first. If more than 20% of bars don't add up, the file is read again with the model, and the result with more correct bars is kept. The card says which one was used.
- **Always**: every file is read with the model. Use this for your copyist's parts.
- **Never**: the model is ignored.

The model runs on the GPU through WebGPU where the browser supports it, otherwise on the CPU through WASM.

## What was tested here, and what wasn't

Tested in this repository's cloud session, which has no GPU and couldn't install PyTorch:
- The MUSCIMA++ conversion: 2,112 / 246 / 239 tiles with 101k / 13k / 12k boxes. Boxes were checked visually.
- The synthetic generator, checked visually.
- The merge step.
- The staff finder: it found all 68 of 68 staves on 10 MUSCIMA++ pages.
- Copyist tiling, pre-labelling and finalize, run with a stand-in ONNX model.
- The accuracy report: its self-check gives 100%, and the ONNX path was also exercised.
- The converter's model path, tested in headless Chromium with the real ONNX Runtime Web 1.31.1 and a stand-in model:
  - the model loads and is still there after a page reload;
  - the 1024×320 tiles map back to the correct page positions;
  - all three modes work;
  - perfect detections derived from the test pages produce exactly the same music as the print rules on all 16 bars, covering clefs, key/time/clef changes, tuplets, grace notes, ties, slurs, dynamics, hairpins, two voices and one-line percussion.

**Not tested yet:** the YOLOX training, evaluation and export cells. They need a GPU and PyTorch, so the first Kaggle run is their first real run. If a cell fails, send me the error from the notebook log.

## License

MUSCIMA++ and CVC-MUSCIMA are licensed CC BY-NC-SA 4.0. Any model trained on them is for **non-commercial** use, must credit those datasets, and must be shared under the same terms. Petaluma is under the SIL Open Font License, and YOLOX is Apache 2.0.
