# BrainSimulation_Decoder
# BrainSim Decoder

An interactive Colab tool for fMRI-to-image decoding using real 7T neuroimaging data from the [Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/) / [Algonauts 2023](http://algonauts.csail.mit.edu/) Challenge.

---

## What it does

Given an input image, the pipeline:

1. Encodes it into a CLIP ViT-L/14 embedding
2. Predicts the whole-cortex fMRI response (~39k vertices, LH + RH) via ridge regression trained on real NSD trials
3. Decodes predicted fMRI back into CLIP space
4. Renders an approximate retinotopic visual field map of predicted activation
5. Retrieves the top-5 most similar training images via decoded CLIP similarity
6. Generates a natural language caption from the decoded representation (BLIP)
7. Synthesizes an audio chord from the decoded embedding

Optionally: GPU image reconstruction via Kandinsky 2.2 unCLIP (requires A100/T4).

---

## Data

Subject 01 from NSD, in Algonauts 2023 format:

```
subj01/
  training_split/
    training_fmri/
      lh_training_fmri.npy
      rh_training_fmri.npy
    training_images/
```

Expected at: `/content/drive/MyDrive/Colab Notebooks/Algonauts Files/subj01`

---

## Setup

```bash
pip install open-clip-torch scikit-learn diffusers>=0.25.0 transformers accelerate soundfile
```

Open `BrainSim_Decoder.ipynb` in Google Colab (T4 or A100). Mount Drive, set `DATA_ROOT`, and run all cells. First run encodes all training images with CLIP (~10 min); subsequent runs load from cache (<30 sec).

---

## Related work

- [Natural Scenes Dataset](https://naturalscenesdataset.org/) — Allen et al. (2022)
- [Algonauts 2023 Challenge](http://algonauts.csail.mit.edu/) — Gifford et al. (2023)
- [MindEye](https://medarc-ai.github.io/mindeye/) — Scotti et al. (2023)

---

## Author

Aayush Gandhi
