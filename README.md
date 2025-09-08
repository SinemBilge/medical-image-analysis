# 🫀 Medical Image Analysis
The focus is on **left ventricle (LV)**, **right ventricle (RV)**, and **myocardium (MYO)** segmentation, feature map analysis, and robustness evaluation under brightness transformations.
## 📂 Project Structure
Notebooks/
├── left-ventricle.ipynb #LV related transformations
├── right-ventricle.ipynb #RV related transformations
├── Myo.ipynb #MYO relared transformations
├── feature_dice_comparison.ipynb #Heatmaps
└── *_data_files/ # text/CSV results from experiments
models/
├── heart_monai-16-4-8_all_0_best.pt
└── heart_monai-8-4-4_all_0_best.pt
src/
├── data.py
├── evaluation.py
├── feature_dice_analysis.py
├── feature_plots.py
├── hooks.py #Model' Layers, feature map extraction
├── matcher.py
├── model.py #visualize results before adjustments
└── utils.py
Task500_ACDC/
└── imagesTr/ 
