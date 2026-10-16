# 🚗 Vehicle Classifier

A Streamlit web app that identifies the **make and model of a car** from a photo. It uses a **MobileNetV2** network fine-tuned with transfer learning on a 48-class dataset of vehicles common on Pakistani roads (Toyota, Honda, Suzuki, Daihatsu, KIA, FAW). Upload one or more images, or take a photo with your camera, and get the predicted model with a confidence score.

**Live demo:** _add your deployed link here_

## Features

- Upload multiple images at once (JPG, JPEG, PNG, BMP, WEBP) or use the camera
- Top-K class probabilities as a bar chart (adjustable, 1 to 10)
- Adjustable confidence threshold: low-confidence predictions are shown as "Uncertain" instead of a wrong guess
- Batch summary table with CSV download of all predictions
- Automatic phone-photo rotation (EXIF) fix
- Runs on CPU, no GPU needed

## Classes (48)

Daiatsu (Core, Hijet, Mira) · FAW (V2, XPV) · Honda (BRV, City Aspire, City 1994/2000, Civic 1994/2005/2007/2015/2018, Grace, Vezel) · KIA Sportage · Suzuki (Alto 2007/2019/Japan 2010, Carry, Cultus 2018/2019, Every, Highroof, Kyber, Liana, Margalla, Mehran, Swift, Wagon R 2015) · Toyota (Aqua, Axio, Corolla 2000/2007/2011/2016, Fortuner, Hiace 2000/2012, Land Cruiser, Passo, Prado, Premio, Prius, Vigo, Vitz, Vitz 2010)

The full ordered list is in [`class_names.json`](class_names.json).

## Model and training

- **Architecture:** MobileNetV2 (ImageNet pretrained) with a new classifier head (Dropout 0.3 + Linear, 48 outputs)
- **Data:** 2,631 train / 465 validation (stratified 15% split) / 751 test images. Class imbalance is 20.8x (largest vs smallest class)
- **Class balancing:** `WeightedRandomSampler`, which reduces the imbalance to 1.84x per epoch
- **Augmentation:** Albumentations, stronger for minority classes
- **Training:** two stages: (1) head only with the backbone frozen, 5 epochs; (2) fine-tuning the upper backbone blocks with early stopping. Label smoothing is used, so confidences are naturally a bit lower
- **Input:** 224 x 224 RGB, ImageNet mean/std normalization
- **Model selection:** by validation macro-F1; the test set was never used for training decisions

### Results (held-out test set, 751 images)

| Experiment | Balancing | Augmentation | Accuracy | Balanced acc. | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| A | none | none | 0.9041 | 0.8459 | 0.8561 |
| B | balanced | none | 0.8935 | 0.8411 | 0.8439 |
| **C (deployed)** | **balanced** | **Albumentations** | **0.9108** | **0.8749** | **0.8792** |

Augmentation reduced test errors from 80 to 67 (26 fixed, 13 newly broken). The error rate on minority classes dropped from 19.1% to 14.3%. McNemar's test gives p = 0.053, so the improvement is promising but not statistically significant at the 5% level.

The full experiment is in the notebook `Deep_Learning_Assigment.ipynb`.

## Project structure

```
├── app.py                          # Streamlit app
├── requirements.txt                # Python dependencies
├── vehicle_mobilenetv2_final.pt    # Trained weights (state_dict, experiment C)
├── class_names.json                # Class names, in training order
├── Deep_Learning_Assigment.ipynb   # Training and evaluation notebook
└── README.md
```

> The weights and `class_names.json` must come from the **same training run**. If they don't match, the app shows a clear error when loading.

## Run locally

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv DL
   source DL/bin/activate        # Windows: DL\Scripts\activate
   ```

3. **Install dependencies.** On a machine without an NVIDIA GPU, install the small CPU-only PyTorch first:

   ```bash
   pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Start the app**

   ```bash
   streamlit run app.py
   ```

5. Open **http://localhost:8501** in your browser.

If the model files are not next to `app.py`, you can upload them from the sidebar instead.

## Usage

1. Open the app and wait for the sidebar to show **Model ready · 48 classes**.
2. Use the **Upload images** tab (or the **Camera** tab) to add pictures.
3. Read the predicted model, confidence, and top-K probabilities.
4. Adjust **Minimum confidence** in the sidebar to control when the app says "Uncertain".
5. For several images, download the **Batch summary** as a CSV.

## Deployment

**Streamlit Community Cloud**

1. Push `app.py`, `requirements.txt`, `vehicle_mobilenetv2_final.pt`, and `class_names.json` to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and select the repo.
3. Set the main file to `app.py` and click **Deploy**.

**Hugging Face Spaces:** create a Space with the Streamlit SDK and upload the same files.

To avoid downloading the large CUDA build of PyTorch on hosts without a GPU, add this as the first line of `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

## Limitations

- The model only knows the 48 classes above. A different vehicle will still get a best guess, which is why the confidence threshold exists.
- Some classes have few images (as few as 11 training images), and several are visually close (for example different Corolla or Civic generations), so these are the most likely to be confused.
- The augmentation improvement over the non-augmented model is not statistically significant (p = 0.053).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `size mismatch for classifier.1.weight` | The weights are not from the trained model (for example, the stock ImageNet checkpoint with 1000 outputs, `mobilenet_v2-b0353104.pth`). Use `vehicle_mobilenetv2_final.pt` from the same run as `class_names.json`. |
| "Weights not found" error | Put the `.pt` file and `class_names.json` next to `app.py`, or upload them in the sidebar. |
| `No space left on device` during install | Install the CPU-only PyTorch build (see above) and use `--no-cache-dir`. |
| `st.bar_chart(horizontal=True)` raises a `TypeError` | Upgrade Streamlit: `pip install --upgrade streamlit`. |
| Browser does not open automatically in WSL | Open `http://localhost:8501` manually. |

## Tech stack

Python, PyTorch, torchvision, Albumentations, Streamlit, pandas, Pillow

## License

_Add your license here (for example, MIT)._
