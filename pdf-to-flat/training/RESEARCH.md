# Handwritten music recognition: research and recommendation (Step 1)

Checked 2026-11-20. Licenses and versions change, so re-check them before training or shipping.

## Recommendation in one paragraph

Train a small **Apache-2.0 object detector (YOLOX-Nano or YOLOX-Tiny)** on staff-sized crops to find handwritten symbols. Run it in the browser with **ONNX Runtime Web** (WebGPU, with a WASM fallback). Keep the converter's existing staff finding, pitch-from-position logic, voice/tuplet/tie handling and the "beats must add up" check. The model replaces only the hand-written shape rules (notehead, rest, accidental, clef and similar tests) on pages that look handwritten. Train on **MUSCIMA++** plus **synthetic pages engraved in the handwriting-style Petaluma font** if the tool stays non-commercial. If it might ever be commercial, train on Petaluma synthetic data plus your own annotated pages only. Expect a large improvement over rule-based reading of handwriting, but not perfection: published detectors on MUSCIMA++ report roughly 60–80+% mAP, so the existing flag-the-bar review step stays essential.

## Datasets

| Source | What it is | License | Usable for |
|---|---|---|---|
| [MUSCIMA++](https://ufal.mff.cuni.cz/muscima) (built on CVC-MUSCIMA) | 140 handwritten pages, 91,254 annotated symbols with relationships (noteheads, stems, beams, rests, accidentals, clefs and more). Images must be fetched from CVC-MUSCIMA separately. | CC BY-NC-SA 4.0 ([project page](https://ufal.mff.cuni.cz/muscima), [tools repo](https://github.com/hajicj/muscima)) | Non-commercial use with attribution. Share-alike: treat the trained model as falling under the same terms. |
| [HOMUS](https://grfia.dlsi.ua.es/homus/) | About 15,000 isolated pen-drawn symbols (online strokes) | Not stated; listed as unknown in the [OMR-Datasets index](https://awesome.ecosyste.ms/projects/github.com%2Fapacha%2Fomr-datasets) | Classifier pre-training at most, and only after asking the authors |
| Synthetic pages in [Petaluma](https://www.smufl.org/fonts/) (Steinberg, handwritten "Real Book" style SMuFL font) | Render generated MusicXML in a handwriting-style font, then add elastic distortion, stroke jitter, ruled-paper background, skew and blur. Bounding boxes come for free. | SIL Open Font License | Any use, including commercial |
| Smashcima ([PyPI](https://pypi.org/project/smashcima/)) | Synthesizer of handwritten music images from MUSCIMA++ symbol masks | Code: Apache 2.0. Generated images inherit MUSCIMA++'s non-commercial terms (stated for its predecessor [Mashcima](https://pypi.org/project/mashcima/)). | Non-commercial; good for volume and variety |
| Your own pages | Annotate 20–50 pages of the handwriting you actually need to read | Yours | Any use; matters most for accuracy on a specific copyist |

Relevant research: [Mayer et al., 2026](https://arxiv.org/abs/2606.09479) report that realistic synthetic manuscript data improves recognition on diverse real manuscripts, but some real in-domain data remains necessary. That supports mixing synthetic data with a small set of real annotated pages.

## Model approach

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| (a) Symbol detector, then the existing assembly logic | Plugs straight into the current pipeline (heads with positions, stems, beams, rests, accidentals). Positions feed the detection overlay. Errors are local and visible. Works with small datasets plus synthetic data. | Needs rules to connect symbols, which the converter already has | **Recommended** |
| (b) End-to-end staff-to-sequence model | No assembly rules | Needs large transcribed corpora; gives no symbol positions for the overlay; errors are hard to explain; usually larger models | Not now |
| (c) Hybrid of (a) for symbols and (b) for rhythm checking | Highest potential | Twice the work | Later, if (a) plateaus |

### Detector choice and licensing

- Ultralytics YOLO models and trained weights are AGPL-3.0 unless you buy a commercial license ([Ultralytics license](https://www.ultralytics.com/license)).
- **YOLOX**, RT-DETR and RF-DETR (base sizes) are Apache 2.0 ([overview](https://www.lightly.ai/blog/best-ultralytics-alternatives-in-2026), [RF-DETR](https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/)). These are vendor or third-party sources, so confirm each repository's LICENSE file.
- Pick YOLOX-Nano (about 0.9M parameters) or YOLOX-Tiny (about 5M). Both export cleanly to ONNX, and quantized they come to roughly 1–10 MB, within the 25 MB budget. The transformer detectors are heavier for in-browser use.
- Run on staff-band crops at the converter's normalized scale (staff space ≈ 22 px), tiled to about 1024×256. Small symbols such as dots and flags need that resolution.

### Proposed classes

These map one-to-one onto the converter's existing event fields. MUSCIMA++ classes can be merged into them.

- Noteheads: filled, half, whole, grace.
- Stem, beam, flag (8th / 16th / 32nd), augmentation dot, ledger line.
- Rests: whole, half, quarter, 8th, 16th.
- Accidentals: sharp, flat, natural, double sharp, double flat.
- Clefs: G, F, C, percussion.
- Time signature digits 0–9, common time, cut time.
- Barline, repeat dots, tie/slur arc, hairpin.
- Dynamics letters (p, m, f, s, z, r).
- Tuplet digits are covered by the time-signature digit classes, using their position relative to beams.

## Browser runtime

- `onnxruntime-web` latest is 1.31.1, published 2026-10-26 ([npm registry](https://npmjs.com/package/onnxruntime-web)). Pin that exact version from jsDelivr.
- Load `ort.webgpu.min.js` for the WebGPU execution provider, which ships in current Chrome and Edge ([ORT WebGPU docs](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)), and fall back to WASM elsewhere.
- Lazy-load the model only when a page looks handwritten or the user turns on "Handwritten".

## Where training can run

This cloud container has 4 CPUs, 15 GB RAM, no GPU and no PyTorch. It can build the data pipeline, the synthetic generator and the browser integration, and it can run a smoke test of training. Real training needs a GPU, such as a Colab/Kaggle notebook, your own machine, or a cloud VM.

As a rough estimate (not measured), YOLOX-Tiny on about 150 real pages plus about 5,000 synthetic staff crops should take a few hours on one modern GPU.

## Decisions needed before Step 2

1. **Commercial or non-commercial use?** This decides whether MUSCIMA++ and Smashcima data can be used.
2. **Where should training run?** Your GPU or a hosted notebook. I'll provide scripts for either.
3. **Whose handwriting matters most?** If it's one copyist or a set of school parts, annotating 20–50 of those pages will help more than anything else.
