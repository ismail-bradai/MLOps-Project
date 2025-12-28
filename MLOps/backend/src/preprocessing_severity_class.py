"""
preprocessing_severity_class.py
Classe pour le preprocessing des données de Sévérité des Accidents (US Accidents)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class PreprocessingSeverity:
    """
    Classe de preprocessing pour la détection de sévérité des accidents.
    """
    
    def __init__(self, data_filename='severity.csv', test_size=0.2, random_state=42):
        # Chemins
        if os.path.exists('/app/processors'):
            self.base_dir = '/'
            self.data_path = os.path.join('/data', data_filename)
            self.processor_dir = '/app/processors'
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_path = os.path.join(self.base_dir, 'data', data_filename)
            self.processor_dir = os.path.join(self.base_dir, 'notebooks', 'processors')
            
        self.test_size = test_size
        self.random_state = random_state
        
        # Processeurs
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_names = {}
        self.smote_config = {'applied': False}
        
        # Configuration des colonnes
        self.target = 'Severity'
        
        # Colonnes à utiliser / conserver (based on standard US Accidents models)
        self.numerical_cols = [
            'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 
            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
        ]
        
        self.boolean_cols = [
            'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
            'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 
            'Traffic_Signal', 'Turning_Loop'
        ]
        
        self.categorical_cols = [
            'Weather_Condition', 'Sunrise_Sunset', 'Wind_Direction', 'Side'
        ]
        
        # Stats
        self.stats = {}

    def load_processors(self):
        """Charge les processeurs existants"""
        processor_files = {
            'scaler': 'scaler.pkl',
            'label_encoders': 'label_encoders.pkl',
            'feature_names': 'feature_names.pkl'
        }
        
        loaded = {}
        for name, filename in processor_files.items():
            filepath = os.path.join(self.processor_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        loaded[name] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if 'scaler' in loaded: self.scaler = loaded['scaler']
        if 'label_encoders' in loaded: self.label_encoders = loaded['label_encoders']
        if 'feature_names' in loaded: self.feature_names = loaded['feature_names']
        
        return len(loaded)

    def _engineer_date_features(self, df):
        """Extraction des features temporelles"""
        if 'Start_Time' in df.columns:
            df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
            
            df['year'] = df['Start_Time'].dt.year
            df['month'] = df['Start_Time'].dt.month
            df['day'] = df['Start_Time'].dt.day
            df['hour'] = df['Start_Time'].dt.hour
            df['weekday'] = df['Start_Time'].dt.weekday
            
            # Drop original date
            # df = df.drop('Start_Time', axis=1) # Keep for now if needed, or drop later
        return df

    def preprocess_inference(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Méthode principale pour nettoyer un input d'inférence"""
        df = df_input.copy()
        
        # 1. Date Features
        df = self._engineer_date_features(df)
        
        # 2. Boolean Conversion
        for col in self.boolean_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool).astype(int)
            else:
                df[col] = 0 # Default to False if missing
                
        # 3. Categorical Encoding
        if 'categorical_features' in self.feature_names:
            expected_cats = self.feature_names['categorical_features']
        else:
            # Fallback based on known usage
            expected_cats = self.categorical_cols
            
        for col in expected_cats:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unknown labels
                df[col] = df[col].astype(str).apply(lambda x: x if x in set(le.classes_) else le.classes_[0])
                df[col] = le.transform(df[col])
            elif col not in df.columns:
                # Add missing categorical column with default value (0)
                df[col] = 0
                
        # 4. Numerical Scaling
        # Use scaler's expected features if available to avoid mismatches
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_nums = self.scaler.feature_names_in_.tolist()
        elif 'numerical_features' in self.feature_names:
            expected_nums = self.feature_names['numerical_features']
        else:
            expected_nums = self.numerical_cols
            
        # Ensure columns exist
        for col in expected_nums:
            if col not in df.columns:
                df[col] = 0.0
        
        if self.scaler:
            try:
                df[expected_nums] = self.scaler.transform(df[expected_nums])
            except Exception as e:
                print(f"⚠️ Scaling warning: {e}")
                # Fallback: try to scale intersection
                valid_cols = [c for c in expected_nums if c in df.columns]
                if valid_cols:
                    df[valid_cols] = self.scaler.transform(df[valid_cols])
            
        # 5. Return expected columns in order
        if 'all_features' in self.feature_names:
            expected_features = self.feature_names['all_features']
        else:
            # Fallback list combining all we know
            expected_features = expected_nums + expected_cats + self.boolean_cols + ['year', 'month', 'day', 'hour', 'weekday']
            
        # Ensure all features exist
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
                
        return df[expected_features]

    # ... Training methods omitted for brevity as we focus on fixing the inference pipeline first ...
    # But for class completeness if the user runs training, we'd need them.
    # Given the urgent fix request, we prioritize the structure used by inference_preprocessor.