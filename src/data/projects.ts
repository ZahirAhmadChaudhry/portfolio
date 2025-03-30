
export const projects = [
  {
    id: 'cryoet-object-identification',
    title: 'CryoET Object Identification',
    description: 'Developing an advanced deep learning pipeline for detecting protein complexes in 3D cryo-electron tomograms using 3D U-Net architecture.',
    fullDescription: 'This project focuses on developing a sophisticated deep learning pipeline for identifying protein complexes in 3D cryo-electron tomography data. The approach leverages volumetric segmentation techniques to process near-atomic-resolution 3D arrays.',
    image: 'https://images.unsplash.com/photo-1532094349884-543bc11b234d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
    category: 'Computer Vision',
    tags: ['3D U-Net', 'PyTorch', 'Computer Vision', 'Deep Learning', 'Zarr'],
    github: 'https://github.com/ZahirAhmadChaudhry/Cryo-ET-Object-Detection-and-Segmentation-Kaggle',
    challenges: [
      'Processing large 3D volumetric data with near-atomic resolution',
      'Mapping voxel intensities to labeled particle centroids',
      'Handling sparse annotations in tomographic data',
      'Balancing computational resources with model complexity'
    ],
    solutions: [
      'Engineered data processing workflow for zarr files to efficiently handle 3D arrays',
      'Implemented base 3D U-Net architecture for volumetric segmentation',
      'Created hybrid datasets combining real and simulated samples',
      'Optimized model with multi-phase detection approach and 3D refinement techniques'
    ],
    achievements: [
      'Successfully validated approach through proof-of-concept testing',
      'Enhanced training data quality through synthetic data generation',
      'Developed efficient preprocessing pipeline for 3D tomographic data'
    ],
    codeSnippet: `# 3D U-Net Model Definition
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        self.encoder1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(256, 512)
        
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)
        
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoding
        enc1 = self.encoder1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.encoder2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.encoder3(pool2)
        pool3 = self.pool3(enc3)
        
        bottleneck = self.bottleneck(pool3)
        
        # Decoding
        up3 = self.upconv3(bottleneck)
        merge3 = torch.cat((up3, enc3), dim=1)
        dec3 = self.decoder3(merge3)
        
        up2 = self.upconv2(dec3)
        merge2 = torch.cat((up2, enc2), dim=1)
        dec2 = self.decoder2(merge2)
        
        up1 = self.upconv1(dec2)
        merge1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.decoder1(merge1)
        
        final = self.final_conv(dec1)
        return final`,
    technologies: [
      '3D U-Net for volumetric segmentation',
      'PyTorch for deep learning model implementation',
      'Zarr file format for efficient 3D data storage and access',
      'GPU acceleration for training and inference',
      'Python data processing pipeline'
    ]
  },
  {
    id: 'recommendation-system',
    title: 'Recommendation System for eCommerce',
    description: 'Led development of advanced recommendation engine analyzing 100k+ transactions using Graph Neural Networks for product suggestions.',
    fullDescription: 'This project involved developing a sophisticated recommendation system for Carrefour eCommerce, leveraging Graph Neural Networks to analyze over 100,000 transactions and provide personalized product recommendations to users.',
    image: 'https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1172&q=80',
    category: 'Classic ML',
    tags: ['GNN', 'CUDA', 'PyTorch', 'SLURM', 'Recommendation Systems'],
    github: 'https://github.com/ZahirAhmadChaudhry/Recommendation_Systems_Kaggle',
    challenges: [
      'Processing and analyzing 100k+ transaction records efficiently',
      'Building scalable recommendation architecture',
      'Balancing recommendation quality with computational efficiency',
      'Handling sparse user-item interaction data',
      'Implementing effective negative sampling'
    ],
    solutions: [
      'Established robust baseline with frequency-based modeling, achieving Hitrate@10 of 0.33',
      'Enhanced performance to 0.34 by incorporating time-weighted purchase patterns',
      'Improved to Hitrate@10 of 0.36 with customer segmentation and segment-specific weighting',
      'Implemented Neural Graph Collaborative Filtering and LightGCN with negative sampling',
      'Optimized deployment on GPU-accelerated SLURM clusters'
    ],
    achievements: [
      'Achieved final Hitrate@10 of 0.36, outperforming baseline models',
      'Successfully implemented and optimized Graph Neural Network approaches',
      'Developed scalable solution capable of handling growing transaction volumes',
      'Created on-the-fly feature loading mechanisms for sparse data handling'
    ],
    codeSnippet: `# Neural Graph Collaborative Filtering Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, layers, dropout):
        super(NGCF, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Weight initialization
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # GNN layers
        self.layers = nn.ModuleList()
        input_size = embedding_dim
        
        for (i, output_size) in enumerate(layers):
            self.layers.append(NGCFLayer(input_size, output_size, dropout))
            input_size = output_size
            
        # Final prediction layer
        self.predictor = nn.Linear(embedding_dim * (len(layers) + 1), 1)
        
    def forward(self, user_indices, item_indices, adjacency_matrix):
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)
        
        # Process through GNN layers
        all_embeddings = [torch.cat([user_emb, item_emb], dim=0)]
        
        for layer in self.layers:
            layer_emb = layer(all_embeddings[-1], adjacency_matrix)
            all_embeddings.append(layer_emb)
            
        # Concatenate embeddings from all layers
        final_embeddings = torch.cat(all_embeddings, dim=1)
        
        # Split user and item embeddings back
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings, [user_emb.shape[0], item_emb.shape[0]], dim=0
        )
        
        # Get embeddings for specific user-item pairs
        user_embeddings = user_all_embeddings[user_indices]
        item_embeddings = item_all_embeddings[item_indices]
        
        # Element-wise product for prediction
        prediction_input = user_embeddings * item_embeddings
        
        # Final prediction
        prediction = self.predictor(prediction_input)
        return prediction.squeeze()`,
    technologies: [
      'PyTorch for implementing Graph Neural Networks',
      'CUDA for GPU acceleration',
      'SLURM for distributed computation',
      'Python for data processing and feature engineering',
      'Neural Graph Collaborative Filtering (NGCF)',
      'LightGCN for simplified graph convolution'
    ]
  },
  {
    id: 'domain-adaptation',
    title: 'Cross-validation for Un-supervised Domain Adaptation',
    description: 'Developed novel transfer learning methodology combining Subspace Alignment and Optimal Transport for cross-domain image classification.',
    fullDescription: 'This project focused on developing a novel transfer learning methodology for cross-domain image classification, bridging the gap between different domains such as Webcam and DSLR images using techniques like Subspace Alignment and Optimal Transport.',
    image: 'https://images.unsplash.com/photo-1561736778-92e52a7769ef?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
    category: 'Computer Vision',
    tags: ['Optimal Transport', 'Transfer Learning', 'PCA', 'scikit-learn', 'Domain Adaptation'],
    github: 'https://github.com/ZahirAhmadChaudhry/transfer-learning-and-optimal-transport-project-AML',
    challenges: [
      'Bridging domain gaps between Webcam and DSLR images',
      'Developing validation framework within unsupervised constraints',
      'Balancing adaptation quality with computational efficiency',
      'Selecting optimal parameters without access to target labels'
    ],
    solutions: [
      'Implemented dual approach combining Subspace Alignment and Optimal Transport',
      'Engineered reverse validation framework for parameter selection',
      'Utilized CaffeNet features for Subspace Alignment',
      'Applied SURF features with Optimal Transport approach'
    ],
    achievements: [
      'Achieved 100% accuracy using Subspace Alignment with CaffeNet features',
      'Reached 82.8% accuracy with Optimal Transport using SURF features',
      'Significantly outperformed the 43.3% baseline',
      'Successfully developed parameter selection methodology for unsupervised domain adaptation'
    ],
    codeSnippet: `# Subspace Alignment Implementation
import numpy as np
from sklearn.decomposition import PCA

def subspace_alignment(source_data, target_data, n_components=50):
    """
    Performs subspace alignment for domain adaptation
    
    Parameters:
    source_data (array): Source domain data
    target_data (array): Target domain data
    n_components (int): Number of principal components
    
    Returns:
    array: Transformed source data aligned to target domain
    """
    # Learn subspaces using PCA
    source_pca = PCA(n_components=n_components)
    target_pca = PCA(n_components=n_components)
    
    source_pca.fit(source_data)
    target_pca.fit(target_data)
    
    # Get projection matrices
    Xs = source_pca.components_.T  # source subspace basis
    Xt = target_pca.components_.T  # target subspace basis
    
    # Compute alignment matrix
    alignment_matrix = Xs.T @ Xt @ Xt.T
    
    # Project source data into aligned subspace
    source_projected = source_pca.transform(source_data)
    source_aligned = source_projected @ alignment_matrix
    
    return source_aligned

# Parameter Selection using Reverse Validation
def reverse_validation(source_data, source_labels, target_data, param_grid):
    """
    Performs reverse validation for parameter selection
    
    Parameters:
    source_data (array): Source domain data
    source_labels (array): Source domain labels
    target_data (array): Target domain data
    param_grid (dict): Grid of parameters to search
    
    Returns:
    dict: Best parameters
    """
    best_score = -np.inf
    best_params = None
    
    for params in param_grid:
        # Adapt source to target
        source_aligned = subspace_alignment(
            source_data, target_data, 
            n_components=params['n_components']
        )
        
        # Reverse direction: adapt target to source
        target_aligned = subspace_alignment(
            target_data, source_data,
            n_components=params['n_components']
        )
        
        # Train classifier on source and evaluate on adapted target
        clf = SVC(C=params['C'], kernel='linear')
        clf.fit(source_aligned, source_labels)
        
        # Pseudo-label the target data
        target_pseudo_labels = clf.predict(target_aligned)
        
        # Train reverse classifier and evaluate on source
        reverse_clf = SVC(C=params['C'], kernel='linear')
        reverse_clf.fit(target_aligned, target_pseudo_labels)
        reverse_score = reverse_clf.score(source_aligned, source_labels)
        
        if reverse_score > best_score:
            best_score = reverse_score
            best_params = params
    
    return best_params`,
    technologies: [
      'Subspace Alignment for domain adaptation',
      'Optimal Transport for distribution matching',
      'Principal Component Analysis (PCA) for dimensionality reduction',
      'scikit-learn for machine learning algorithms',
      'Python for implementation and evaluation',
      'CaffeNet and SURF features for image representation'
    ]
  },
  {
    id: 'credit-card-approval',
    title: 'Credit Card Approval Classification using SVM',
    description: 'Developed comprehensive machine learning pipeline for credit card approval classification, achieving 0.86+ accuracy and ROC-AUC score above 0.90.',
    image: 'https://images.unsplash.com/photo-1563013544-824ae1b704d3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
    category: 'Classic ML',
    tags: ['SVM', 'Classification', 'scikit-learn', 'Feature Engineering', 'Hyperparameter Optimization'],
    github: 'https://github.com/ZahirAhmadChaudhry/credit-cards-approval-classification-using-SVM',
    challenges: [
      'Handling diverse data types including categorical and numerical features',
      'Dealing with skewed and imbalanced data',
      'Selecting optimal SVM parameters for best performance',
      'Building robust evaluation framework for reliable performance estimates'
    ],
    solutions: [
      'Implemented feature preprocessing pipeline with one-hot encoding and Min-Max scaling',
      'Applied skewness correction for numerical features',
      'Conducted extensive experimentation with multiple SVM variants',
      'Utilized systematic hyperparameter optimization through grid search',
      'Implemented stratified splitting for handling class imbalance'
    ],
    achievements: [
      'Achieved superior performance using Linear SVM with 0.86+ accuracy',
      'Obtained ROC-AUC score above 0.90 on test set',
      'Enhanced model robustness through cross-validation',
      'Established foundation for future improvements in cost-sensitive learning'
    ],
    codeSnippet: `# SVM Model Training and Evaluation Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Define preprocessing pipeline
def create_preprocessing_pipeline(numerical_cols, categorical_cols):
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

# Create full pipeline with SVM
def create_model_pipeline(preprocessor):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True))
    ])
    return pipeline

# Define hyperparameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto', 0.1, 0.01]
}

# Train model with cross-validation
def train_model(X, y, pipeline, param_grid):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return results`,
    technologies: [
      'Support Vector Machines (SVM) for classification',
      'scikit-learn for machine learning pipeline',
      'Feature engineering and preprocessing techniques',
      'Hyperparameter optimization with grid search',
      'Cross-validation for robust evaluation',
      'ROC-AUC and accuracy metrics for performance assessment'
    ]
  },
  {
    id: 'healthcare-hai-risk',
    title: 'Healthcare-Associated Infection Risk Prediction',
    description: 'Designed and implemented an end-to-end machine learning pipeline for predicting Healthcare-Associated Infection risks in ICU settings.',
    image: 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
    category: 'Classic ML',
    tags: ['Healthcare', 'GBM', 'SVM', 'Autoencoder', 'Feature Selection'],
    challenges: [
      'Integrating three complex healthcare data sources with 412 data points',
      'Working with highly dimensional medical data',
      'Collaborating with medical experts to optimize feature selection',
      'Balancing model interpretability with prediction performance'
    ],
    solutions: [
      'Conducted comprehensive literature review on HAI prediction models',
      'Engineered unified dataset by integrating three complex healthcare sources',
      'Collaborated with medical experts to reduce dimensions from 265 to 70 critical indicators',
      'Implemented multiple ML models including Logistic Regression, Autoencoders, SVM, and GBM'
    ],
    achievements: [
      'Achieved 99.9% accuracy and 0.999 R² through cross-validation with Gradient Boosting Machines',
      'Developed interactive clinical decision support system',
      'Facilitated cross-functional collaboration between Medical, Management, and ML experts',
      'Created interpretable risk assessment model for clinical use'
    ],
    codeSnippet: `# Feature Selection and GBM Implementation
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import shap

# Feature selection with domain expertise
def expert_guided_feature_selection(data, expert_feedback):
    """
    Performs feature selection based on expert feedback
    
    Parameters:
    data (DataFrame): Original dataset with all features
    expert_feedback (dict): Dictionary of feature importance from domain experts
    
    Returns:
    DataFrame: Dataset with selected features
    """
    # Filter features based on expert importance threshold
    important_features = [f for f, score in expert_feedback.items() if score > 0.7]
    
    # Keep only the important features plus the target variable
    selected_data = data[important_features + ['target']]
    
    return selected_data

# Train GBM model with optimized parameters
def train_optimized_gbm(X_train, y_train):
    """
    Trains Gradient Boosting Machine with optimized parameters
    
    Parameters:
    X_train (array): Training features
    y_train (array): Training target
    
    Returns:
    object: Trained GBM model
    """
    # Preprocess features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize and train GBM
    gbm = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.8,
        random_state=42
    )
    
    gbm.fit(X_train_scaled, y_train)
    
    return gbm, scaler

# Evaluate model performance with cross-validation
def evaluate_with_cross_validation(X, y, n_folds=5):
    """
    Evaluates model with stratified k-fold cross-validation
    
    Parameters:
    X (array): Features
    y (array): Target
    n_folds (int): Number of folds for cross-validation
    
    Returns:
    dict: Performance metrics
    """
    # Initialize model
    gbm = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        random_state=42
    )
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Calculate metrics using cross-validation
    accuracy = cross_val_score(gbm, X, y, cv=cv, scoring='accuracy')
    r2 = cross_val_score(gbm, X, y, cv=cv, scoring='r2')
    auc = cross_val_score(gbm, X, y, cv=cv, scoring='roc_auc')
    
    # Return aggregated metrics
    metrics = {
        'accuracy_mean': accuracy.mean(),
        'accuracy_std': accuracy.std(),
        'r2_mean': r2.mean(),
        'r2_std': r2.std(),
        'auc_mean': auc.mean(),
        'auc_std': auc.std()
    }
    
    return metrics`,
    technologies: [
      'Gradient Boosting Machines for risk prediction',
      'Support Vector Machines and Autoencoders for comparative analysis',
      'Feature selection with domain expert guidance',
      'Cross-validation for robust model evaluation',
      'Interactive visualization for clinical interpretation',
      'Collaborative model development with medical experts'
    ]
  },
  {
    id: 'nlp-sustainability',
    title: 'NLP Solution for Sustainability Opinion Analysis',
    description: 'Designing and implementing an end-to-end NLP solution for automating opinion analysis in French sustainability discussions.',
    image: 'https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80',
    category: 'NLP',
    tags: ['NLP', 'Opinion Mining', 'BERT', 'Multilingual', 'Web Application'],
    challenges: [
      'Analyzing 300+ pages of multi-speaker transcripts',
      'Identifying organizational paradoxes and tensions in complex discussions',
      'Creating NLP pipeline for multilingual text analysis',
      'Building interactive visualization for cross-report comparison'
    ],
    solutions: [
      'Developed specialized NLP techniques for organizational paradox identification',
      'Created ML pipeline bridging management science theory with computational approaches',
      'Implemented web-based application with interactive visualization capabilities',
      'Collaborated with interdisciplinary team of management researchers',
      'Enhanced model interpretability through transparent classification processes'
    ],
    achievements: [
      'Successfully automated opinion analysis in sustainability discussions',
      'Integrated domain expertise into algorithmic design',
      'Provided interactive result editing capabilities for expert users',
      'Created visualizations for cross-report comparison'
    ],
    codeSnippet: `# Multilingual Opinion Mining with BERT
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd

class MultlingualOpinionAnalyzer:
    def __init__(self, model_name="camembert-base"):
        """
        Initialize the multilingual opinion analyzer
        
        Parameters:
        model_name (str): Name of the pretrained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # positive, negative, neutral
        )
        
        # Set up device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def fine_tune(self, texts, labels, epochs=3):
        """
        Fine-tune the model on domain-specific data
        
        Parameters:
        texts (list): List of input texts
        labels (list): List of corresponding labels
        epochs (int): Number of training epochs
        """
        # Tokenize input texts
        encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Convert labels to tensor
        labels = torch.tensor(labels).to(self.device)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(**encodings, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            
    def analyze_text(self, text, extract_tensions=True):
        """
        Analyze text for opinions and tensions
        
        Parameters:
        text (str): Input text to analyze
        extract_tensions (bool): Whether to extract organizational tensions
        
        Returns:
        dict: Analysis results
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get opinion classification
        logits = outputs.logits
        predictions = torch.nn.functional.softmax(logits, dim=-1)
        sentiment_scores = predictions[0].cpu().numpy()
        
        # Extract tensions if requested
        tensions = []
        if extract_tensions:
            tensions = self._extract_organizational_tensions(text)
            
        # Return analysis results
        results = {
            "sentiment": {
                "positive": float(sentiment_scores[0]),
                "neutral": float(sentiment_scores[1]),
                "negative": float(sentiment_scores[2])
            },
            "overall_sentiment": ["negative", "neutral", "positive"][np.argmax(sentiment_scores)],
            "tensions": tensions
        }
        
        return results
        
    def _extract_organizational_tensions(self, text):
        """
        Extract organizational tensions from text
        
        Parameters:
        text (str): Input text
        
        Returns:
        list: Extracted tensions
        """
        # This would be a more complex implementation in practice
        # Using keyword-based approach as placeholder
        tension_indicators = [
            ("sustainability", "profit"),
            ("short-term", "long-term"),
            ("innovation", "stability"),
            ("centralization", "decentralization"),
            ("control", "autonomy")
        ]
        
        found_tensions = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.lower()
            for t1, t2 in tension_indicators:
                if t1 in sentence and t2 in sentence:
                    found_tensions.append({
                        "type": f"{t1} vs {t2}",
                        "sentence": sentence.strip(),
                        "confidence": 0.85  # Placeholder for actual confidence score
                    })
                    
        return found_tensions`,
    technologies: [
      'BERT/CamemBERT for multilingual text analysis',
      'PyTorch for NLP model implementation',
      'Web application with interactive visualizations',
      'Natural Language Processing techniques',
      'Opinion mining and sentiment analysis',
      'Organizational paradox identification algorithms'
    ]
  }
];
