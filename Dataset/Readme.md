## 📁 Dataset

The dataset used in this project is a sample of the **DeepFake Detection Challenge Dataset** hosted on Kaggle.  
🔗 [Dataset Link](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

### 📦 Dataset Structure

- **Train Data:**  
  Contains **400 videos** in `.mp4` format.

- **Test Data:**  
  Contains **400 videos** in `.mp4` format.

- **Metadata (JSON format):**  
  Accompanies the video files and contains the following details:

  | Key       | Description                                                                 |
  |-----------|-----------------------------------------------------------------------------|
  | `filename` | The filename of the video                                                  |
  | `label`    | Label for classification – `1` for **FAKE**, `0` for **REAL**             |
  | `original` | If the video is fake, this field shows the original source video filename |
  | `split`    | Indicates dataset split – this is always `"train"` in our sample          |

---

> ⚠ **Note:** For this project, we used only a small, manageable portion of the dataset for training, testing, and evaluation due to storage and computational constraints.
