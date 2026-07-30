\# 🧠 Face Recognition AI — MobileNetV2 + Streamlit



A deep learning-based face recognition web application that identifies registered individuals from an uploaded photo, using transfer learning with \*\*MobileNetV2\*\*. If the model isn't confident enough in a prediction, it flags the image as \*\*"Unknown Person"\*\* instead of guessing.



Trained in Google Colab (GPU-accelerated), deployed as an interactive \*\*Streamlit\*\* web app.



\---



\## 🎯 What it does



\- Upload a face photo through the web app

\- The model preprocesses the image and predicts which registered person it is

\- Displays the prediction with a confidence score and full class probabilities

\- If confidence falls below an adjustable threshold, shows \*\*"Unknown Person"\*\* instead of a wrong guess

\- Confidence threshold is adjustable live via a sidebar slider



\---



\## 🖼️ Demo



\*(Add a screenshot or short GIF of the app here once deployed — this is the first thing recruiters look at.)\*



\---



\## 🛠️ Tech Stack



| Category | Technology |

|---|---|

| Language | Python |

| Deep Learning | TensorFlow / Keras |

| Model Architecture | MobileNetV2 (Transfer Learning) |

| Image Processing | Pillow, `tf.image` |

| Numerical Computing | NumPy |

| Visualization | Matplotlib, Seaborn |

| Training Platform | Google Colab (GPU) |

| Dataset Storage | Google Drive |

| Web Framework | Streamlit |

| Development Environment | VS Code |

| Version Control | Git \& GitHub |



\---



\## 📊 Dataset



The dataset consists of facial images of \*\*two registered individuals\*\*, captured across varied conditions to help the model generalize:



\- Different lighting (indoor, outdoor, natural, artificial)

\- Multiple angles (front-facing, slight turns, tilts)

\- Various facial expressions

\- Different backgrounds and locations



| Class | Images |

|---|---|

| Person 1 | 477 |

| Person 2 | 223 |



Split \*\*70% train / 15% validation / 15% test\*\*, using stratified sampling to preserve class balance across all three sets.



\*\*Note on class imbalance:\*\* the dataset is roughly 2:1 across classes. This was addressed using `sklearn.utils.class\_weight.compute\_class\_weight`, which upweights the underrepresented class during training so the model doesn't default to over-predicting the majority class.



\*\*Note on file formats:\*\* source images included a mix of `.jpg` and `.HEIC` (Apple's photo format). HEIC files were converted to JPEG using `pillow-heif` prior to training, since TensorFlow's image decoders don't natively support HEIC.



\---



\## 🏗️ Model Architecture \& Training



Built using \*\*transfer learning\*\* with MobileNetV2 (pretrained on ImageNet), chosen for its strong accuracy-to-compute tradeoff — well suited for a lightweight, deployable classifier.



\*\*Training was done in two phases:\*\*



1\. \*\*Phase 1 — Frozen base:\*\* MobileNetV2's convolutional base was frozen, and only a new classification head (GlobalAveragePooling → Dropout → Dense(64) → Dropout → Dense(softmax)) was trained. This lets the new head adapt quickly without disturbing the pretrained features.

2\. \*\*Phase 2 — Fine-tuning:\*\* The last \~30 layers of MobileNetV2 were unfrozen and trained further at a much lower learning rate (1e-5), to adapt the pretrained features more specifically to these two faces without catastrophic forgetting.



\*\*Data augmentation\*\* (random flips, rotation, zoom, contrast, brightness) was applied during training to improve generalization given the moderate dataset size.



`EarlyStopping` (monitoring validation loss, `restore\_best\_weights=True`) was used in both phases to avoid overfitting and automatically roll back to the best-performing checkpoint.



\### Results



\- \*\*Validation accuracy:\*\* \~95%

\- \*\*Validation loss:\*\* \~0.12–0.14 at best checkpoint

\- Full evaluation on a held-out test set includes precision, recall, F1-score, and a confusion matrix (see the training notebook for the complete classification report)



\*(Fill in final test accuracy/loss from your Cell 12 output here once confirmed.)\*



\---



\## 📁 Project Structure



```

FaceRecognitionApp/

├── model/

│   ├── face\_recognition\_model.keras   # trained model

│   └── class\_names.json               # class label order

├── face\_recognition\_training.ipynb    # full Colab training notebook

├── app.py                             # Streamlit web app

├── requirements.txt

└── README.md

```



\---



\## 🚀 Running Locally



```bash

\# Clone the repo

git clone https://github.com/AlwinSanjai/FaceRecognitionApp.git

cd FaceRecognitionApp



\# Create and activate a virtual environment

python -m venv venv

venv\\Scripts\\activate        # Windows

source venv/bin/activate     # macOS/Linux



\# Install dependencies

pip install -r requirements.txt



\# Run the app

streamlit run app.py

```



The app opens automatically at `http://localhost:8501`. Upload a photo and get a prediction.



\---



\## 🧪 How the "Unknown Person" logic works



Every prediction includes a confidence score (softmax probability of the top class). If that score falls below a configurable threshold (default: 75%), the app reports \*\*"Unknown Person"\*\* instead of forcing a match. This threshold is adjustable live from the sidebar, so you can see how prediction behavior shifts in real time — useful for demonstrating the precision/recall tradeoff of the confidence gate.



\---



\## 🔁 Reproducing the training



The full pipeline — from mounting Google Drive, through HEIC conversion, dataset splitting, class-weight computation, two-phase MobileNetV2 training, and evaluation — is documented step-by-step in \[`face\_recognition\_training.ipynb`](./face\_recognition\_training.ipynb). Open it in Google Colab, mount your own Drive with a similarly structured dataset (`dataset/<person\_name>/\*.jpg`), and run through the cells in order.



\---



\## 🔮 Future Enhancements



\- Recognize more than two individuals

\- Real-time face recognition via webcam

\- Self-service user registration for new faces

\- Store prediction history in a database

\- User authentication for the admin/registration flow

\- Continuous deployment pipeline on a cloud platform



\---



\## 📌 Resume Summary



> Developed a deep learning-based face recognition web application using TensorFlow, MobileNetV2, and Streamlit. Trained a transfer learning model in Google Colab (two-phase training with class-weighted loss to handle dataset imbalance) to recognize registered individuals from uploaded images, implemented confidence-based unknown-face detection, and deployed the solution through an interactive web interface. Demonstrates skills in computer vision, deep learning, transfer learning, model evaluation, and full-stack AI application development.



\---



\## 📄 License



\*(Add a license if you want this repo to be reusable by others, e.g. MIT.)\*

