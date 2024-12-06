
 # Credit Card Transactions’ Anomaly Detection Using Semi Self-Supervised Learning  

## Overview  
This project focuses on building a robust anomaly detection model to identify fraudulent credit card transactions in highly imbalanced financial datasets. By leveraging a semi self-supervised learning approach, the model effectively distinguishes between normal and anomalous transactions, enhancing fraud prevention while minimizing false positives.  

## Features  
- **Semi Self-Supervised Learning:** Contrastive learning is employed to train a neural network, generating embeddings that clearly differentiate normal and anomalous transactions.  
- **Dimensionality Reduction:** Principal Component Analysis (PCA) reduces the feature space, highlighting critical attributes and eliminating noise.  
- **Anomaly Scoring:** Distance-based methods such as k-Nearest Neighbors (k-NN) and DBSCAN are used to score and classify transactions based on their embedding similarities.  
- **High Precision:** Achieved a precision score of 0.8, optimizing the balance between fraud detection and false alarms.  

## Objectives  
- Build a scalable and efficient model for anomaly detection in financial transaction datasets.  
- Address data imbalance and enhance model focus on relevant features.  
- Provide actionable insights for real-time fraud prevention systems.  

## Technology Stack  
- **Programming Language:** Python  
- **Libraries:** PyTorch, Scikit-learn, NumPy, Pandas  
- **Algorithms:** Contrastive Learning, PCA, k-NN, DBSCAN  

## Dataset  
The project utilizes a financial transactions dataset, with features such as transaction amount, timestamps, and labels indicating fraudulent or normal transactions.  

## How to Run  
1. **Clone the Repository:**  
   ```bash  
   git clone <repository-url>  
   cd <repository-directory>  
   ```  

2. **Install Dependencies:**  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run the Model Training:**  
   ```bash  
   python train_model.py  
   ```  

4. **Evaluate the Model:**  
   ```bash  
   python evaluate_model.py  
   ```  

## Results  
- Achieved a precision score of 0.8, ensuring reliable fraud detection with reduced false positives.  
- Demonstrated the model’s potential for real-world financial fraud detection systems.  

## Future Scope  
- Integrate the model into a real-time transaction monitoring system.  
- Explore additional unsupervised methods for improving detection of rare anomalies.  

## Contributing  
Contributions are welcome! Please feel free to submit a pull request or raise an issue for any bugs or feature enhancements.  

