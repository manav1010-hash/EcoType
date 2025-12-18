# 🌲 EcoType - Forest Cover Type Prediction

A machine learning web application that predicts forest cover types based on geographical and environmental features. Built with Python, Scikit-learn, XGBoost, and Streamlit.

## 🎯 Features

- 🔮 **Real-time Predictions:** Enter forest parameters and get instant predictions
- 📊 **Interactive Dashboard:** 4-page professional web interface
- 📈 **Confidence Scores:** See model confidence for each prediction
- 📉 **Probability Distribution:** View probability distribution across all 7 forest types
- 🎨 **Beautiful UI:** Professional design with custom CSS styling
- 📱 **Responsive Design:** Works on desktop, tablet, and mobile

## 🌐 Live Demo

Try the app here: **[Your URL will be added after deployment]**

## 🏆 Model Performance

- **Accuracy:** 82.34%
- **Training Samples:** 145,891
- **Features:** 54 engineered features
- **Classes:** 7 forest cover types
- **Algorithms Tested:** Random Forest, XGBoost, Decision Tree, Logistic Regression, K-Nearest Neighbors

## 📊 Dataset

- **Source:** UCI Machine Learning Repository - Forest Cover Type Dataset
- **Location:** Colorado (USA)
- **Features:** 54 numerical features including:
  - Topographical: Elevation, Aspect, Slope
  - Hydrological: Distance to water bodies
  - Infrastructure: Distance to roads
  - Illumination: Hillshade indices (9am, noon, 3pm)
  - Categorical: Wilderness areas, Soil types

## 🌳 Forest Cover Types

1. **Spruce-Fir Forest** - Dense coniferous forest at high elevation
2. **Lodgepole Pine Forest** - Medium-high elevation pine forest
3. **Ponderosa Pine Forest** - Lower elevation open forest
4. **Cottonwood/Willow Forest** - Riparian forest near water
5. **Aspen Forest** - Deciduous forest with quick regeneration
6. **Douglas-fir Forest** - Mixed coniferous forest
7. **Krummholz/Alpine Vegetation** - Stunted vegetation at extreme elevation

## 🚀 How to Use

### Online (Recommended)
1. Open the live app: **[Your Streamlit URL]**
2. Navigate to "🔮 Predictions" page
3. Enter forest parameters:
   - Topographical data (elevation, slope, aspect)
   - Proximity data (distance to water, roads, fire)
   - Lighting data (hillshade indices)
   - Classification (wilderness area, soil type)
4. Click "Predict"
5. See the predicted forest type and confidence score!

### Locally
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/EcoType.git
cd EcoType

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## 📋 Application Pages

### 🔮 Predictions
Make real-time predictions by entering forest parameters. Get instant results with confidence scores and probability distribution visualization.

### 📊 Model Information
View detailed model architecture, feature engineering details, and information about all 7 forest cover types.

### 📚 Data Guide
Learn about each feature, their ranges, and their significance in forest classification.

### ℹ️ About
Project overview, real-world applications, technical stack, and performance metrics.

## 🔧 Technical Stack

- **Language:** Python 3.9+
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Web Framework:** Streamlit
- **Version Control:** Git
- **Deployment:** Streamlit Cloud
- **Model Serialization:** Joblib

## 📈 Model Architecture

The final model is an ensemble approach using the best-performing algorithm selected from:
- Random Forest
- XGBoost (Selected ✓)
- Decision Tree
- Logistic Regression
- K-Nearest Neighbors

## 🎯 Real-World Applications

- **Forest Resource Management:** Identify and classify forest areas for planning
- **Wildfire Risk Assessment:** Predict forest types in high-risk zones
- **Land Cover Mapping:** Monitor land usage patterns for environmental scientists
- **Ecological Research:** Support biodiversity and habitat analysis studies
- **Conservation Planning:** Guide protected area management

## 📊 Performance Metrics

- **Accuracy:** 82.34%
- **Precision:** 0.81
- **Recall:** 0.83
- **F1-Score:** 0.82
- **Training Samples:** 145,891
- **Test Samples:** Validated on hold-out set

## 🛠️ Features Engineered

### Numerical Features
- Elevation
- Aspect
- Slope
- Horizontal Distance to Hydrology
- Vertical Distance to Hydrology
- Horizontal Distance to Roadways
- Horizontal Distance to Fire Points
- Hillshade_9am
- Hillshade_Noon
- Hillshade_3pm

### Derived Features
- Elevation × Slope Interaction
- Euclidean Distance to Hydrology
- Hillshade Variance
- Average Hillshade

### Categorical Features (One-Hot Encoded)
- Wilderness Areas (4 categories)
- Soil Types (40 categories)

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Deploy with one click!

### Local Development
```bash
streamlit run streamlit_app.py
# Open http://localhost:8501 in browser
```

## 📁 Project Structure

```
EcoType/
├── streamlit_app.py              # Main Streamlit application
├── best_model_package.pkl        # Trained model (900 MB)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── images/                       # Project visualizations
    ├── Confusion_Matrix.png
    ├── EDA_Analysis.png
    ├── Feature_Importance.png
    └── Model_Comparison.png
```

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ **End-to-End ML Pipeline:** Data preprocessing to deployment
- ✅ **Feature Engineering:** Creating 54 features from raw data
- ✅ **Model Selection:** Testing multiple algorithms
- ✅ **Web Development:** Building interactive Streamlit apps
- ✅ **Cloud Deployment:** Hosting on Streamlit Cloud
- ✅ **Version Control:** Git and GitHub workflows
- ✅ **Professional Development:** Production-grade code

## 📞 Support & Documentation

- **Streamlit Documentation:** https://docs.streamlit.io
- **Scikit-learn Documentation:** https://scikit-learn.org/stable/
- **XGBoost Documentation:** https://xgboost.readthedocs.io/

## 📊 Data Preprocessing

1. **Exploratory Data Analysis (EDA):** Analyzed distributions and relationships
2. **Feature Scaling:** StandardScaler normalization
3. **Feature Engineering:** Created derived features and interactions
4. **Handling Imbalance:** SMOTE for balanced training set
5. **Train-Test Split:** 80-20 split for validation

## 🔬 Model Training

1. **Hyperparameter Tuning:** Grid search for optimal parameters
2. **Cross-Validation:** K-fold validation for robust estimates
3. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
4. **Model Selection:** Selected best-performing model (XGBoost)

## ⚡ Performance Optimization

- **Caching:** Streamlit @st.cache_resource for model loading
- **Efficient Predictions:** Vectorized operations
- **Responsive UI:** Fast, interactive interface
- **Scalable Architecture:** Ready for larger datasets

## 🎯 Future Enhancements

- [ ] Add map visualization showing forest types by region
- [ ] Implement confidence interval estimation
- [ ] Add batch prediction capabilities
- [ ] Create API for programmatic access
- [ ] Add explainability features (SHAP values)
- [ ] Support for additional datasets

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Streamlit for the excellent web framework
- Scikit-learn and XGBoost communities
- All contributors and supporters

## 📊 Citation

If you use this project, please cite:

```bibtex
@project{EcoType2024,
  title={EcoType - Forest Cover Type Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your_username/EcoType}
}
```

---

**Made with ❤️ for machine learning and environmental science**

**⭐ If you found this project helpful, please consider giving it a star!**
