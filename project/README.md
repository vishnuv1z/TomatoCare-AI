# TomatoCare AI

TomatoCare AI is a Streamlit prototype that classifies an uploaded tomato leaf image into one of 10 predefined classes with a trained TensorFlow/Keras model. The interface displays the predicted class, confidence, class probabilities, and general disease information.

Predictions are automated estimates and are not a professional agricultural diagnosis. Use a qualified local expert for decisions about crop treatment.

## Project files

```text
tomato-disease-project/
├── app.py
├── tomato_disease_model.keras   # Add your trained model here
├── requirements.txt
└── README.md
```

The model file is not included. Place `tomato_disease_model.keras` in the same folder as `app.py` before running predictions.

## Windows setup in VS Code

The provided requirements resolve for native Windows CPU use with Python 3.11. The TensorFlow wheel on native Windows is CPU-oriented; use WSL2 with the VS Code **WSL** extension if you need GPU acceleration. Python 3.10 or 3.11 is a practical starting point. Use a TensorFlow/Keras version compatible with the one that saved your `.keras` model.

1. Open this project folder in VS Code and select a Python 3.11 interpreter.
2. In the integrated PowerShell terminal, create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   For WSL2, use `python3 -m venv .venv` and `source .venv/bin/activate` instead.

3. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Add `tomato_disease_model.keras` beside `app.py`.
5. Start the application:

   ```powershell
   streamlit run app.py
   ```

Streamlit prints the local address in the terminal, usually `http://localhost:8501`.

## Model and preprocessing expectations

- The model input is one RGB image with shape `224 × 224 × 3`.
- The app resizes the image to `224 × 224` and applies `tf.keras.applications.mobilenet_v2.preprocess_input`, which scales pixel values to the range expected by MobileNetV2.
- The training pipeline must have used the same preprocessing and the class order below. Update `CLASS_NAMES` in `app.py` if the model was trained with a different output order.
- The classifier must return 10 scores in this order:

  1. `Tomato___Bacterial_spot`
  2. `Tomato___Early_blight`
  3. `Tomato___Late_blight`
  4. `Tomato___Leaf_Mold`
  5. `Tomato___Septoria_leaf_spot`
  6. `Tomato___Spider_mites_Two-spotted_spider_mite`
  7. `Tomato___Target_Spot`
  8. `Tomato___Tomato_mosaic_virus`
  9. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
  10. `Tomato___healthy`

The model is loaded once with Streamlit's `st.cache_resource`. A missing model, invalid image, unsupported extension, incompatible output, or inference error is reported in the app instead of intentionally terminating the session.

## Evaluation metrics

The Model Information page leaves accuracy, precision, recall, and F1-score as `XX%` placeholders. Replace them with values calculated from a held-out evaluation set; this app does not claim model performance metrics.