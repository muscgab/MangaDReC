# Third-party notices

MangaDReC and MangaDReCo include or build on the following projects and data.
Their licenses remain applicable to the corresponding components.

## PaddleOCR and PP-OCRv6

- Project: PaddleOCR
- Source: <https://github.com/PaddlePaddle/PaddleOCR>
- License: Apache License 2.0
- Use: DET/REC architectures and upstream pretrained checkpoints. The released
  DET/REC weights were trained further for this project.

## PaddleOCR2Pytorch

- Project: PaddleOCR2Pytorch
- Source: <https://github.com/frotms/PaddleOCR2Pytorch>
- License: Apache License 2.0
- Use: PyTorch model definitions used by the packaged runtime.
- A copy of its license is retained under
  `runtime/third_party/PaddleOCR2Pytorch/LICENSE` in each model package.

## manga-ocr

- Project: manga-ocr
- Source: <https://github.com/kha-white/manga-ocr>
- License: Apache License 2.0
- Use: Initial OCR transcription for text extracted from a private manga
  collection. Terra and Luna subsequently repaired and filtered those
  transcripts. Neither manga-ocr code nor weights are bundled here.

## JESC

- Dataset: Japanese-English Subtitle Corpus (JESC)
- Source: <https://nlp.stanford.edu/projects/jesc/>
- License: Creative Commons Attribution-ShareAlike 4.0 International
- Use: Part of MangaDReCo B's Japanese semantic pretraining corpus.
- JESC text is not redistributed in the release packages.

## Training corpus disclosure

The private manga corpus was transcribed with manga-ocr and then repaired and
filtered by Terra and Luna. Only derived model weights are distributed. The
private manga images, OCR transcripts, cleaned text corpus, synthetic training
images, and generator assets are not included in the public release.

