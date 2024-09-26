# Speech Emotion Recognition (SER)

## 📄 Project Overview

This **Speech Emotion Recognition (SER)** project identifies the emotional state in spoken audio using machine learning techniques. The model analyzes speech features such as MFCC (Mel-frequency cepstral coefficients) to predict emotions like happiness, sadness, anger, and more. It achieved a **98% accuracy** without hyperparameter tuning or cross-validation.

## 🧠 Features

- **Emotion Detection**: Recognizes emotions such as happy, sad, angry, neutral, etc.
- **High Accuracy**: Achieved 98% accuracy using a classification model.
- **Audio Processing**: Utilizes MFCC and other advanced features.
- **Scalable**: Can be extended to recognize real-time emotions in speech.

## 📊 Dataset

This project uses the Toronto Emotional Speech Set (TESS) dataset, which consists of speech recordings portraying seven emotions
(anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). 
You can download the dataset from Kaggle using the link below: [Toronto Emotional Speech Set (TESS) dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess/data)

TESS Dataset on Kaggle
## 🛠 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Manikandan200355/Zidio_DataScience_Internship.git
    cd speech-emotion-recognition
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # For Windows: `env\Scripts\activate`
    ```

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Training the Model
To train the emotion recognition model, run the following command:
```bash
python train.py
```

### 2. Testing the Model
To test the model performance, run:
```bash
python test.py
```

### 3. Making Predictions
To predict the emotion of a new audio file:
```bash
python predict.py --file path_to_audio_file.wav
```

## 🔧 Preprocessing

- **Audio Features**: Extracts MFCC, Chroma, and Mel-spectrogram features from the audio data.
- **Data Splitting**: The dataset is divided into training and testing sets to train the model and evaluate its performance.

## 🧑‍💻 Model Architecture

- A classification model is used to predict emotions based on extracted audio features.
- **Techniques Used**: MFCC, Chroma, and Mel-spectrogram for feature extraction.
  
## 📈 Results

- **Model Accuracy**: The SER model achieved a remarkable **98% accuracy** without hyperparameter tuning.
- **Confusion Matrix**: Detailed breakdown of prediction success and failure rates for each emotion.

## 🔮 Future Enhancements

- **Cross-Validation**: Implement cross-validation to optimize model performance.
- **Real-time Emotion Recognition**: Extend functionality to detect emotions in real-time speech.
- **Deep Learning Architectures**: Experiment with advanced architectures to further improve results.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Special thanks to my mentor **Chandan Mishra** for guidance and support.
- Libraries: **Librosa** (for audio processing), **sklearn** (for model training and evaluation).
