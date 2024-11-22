import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import json
import pickle



def preprocess_data(df):
    """
    Applique toutes les étapes de prétraitement sur un DataFrame donné.
    
    Paramètres :
        df (pd.DataFrame) : Nouveau DataFrame à prétraiter.
        
    Retourne :
        pd.DataFrame : DataFrame prétraité.
    """
    def encodage_categoriel(colonne_name) :
        unique_value= sorted(df[colonne_name].unique())
        Ordencod_ = OrdinalEncoder(categories=[unique_value])
        df[f'{colonne_name}_Encoded'] = Ordencod_.fit_transform(df[[colonne_name]])
        return df

    # Étape 1 : Encodage de 'optical_code' - Séparer 'optical_code' en colonnes distinctes
    def separate_data_optical_code(value):
        letter = value[0]
        transparency = value[1:4]
        teinte = value[4:7]
        return letter, transparency, teinte
    
    df[['colorimetrie', 'transparency', 'teinte']] = df['optical_code'].apply(
        separate_data_optical_code
    ).apply(pd.Series)
    
    df['colorimetrie_Encoded'], _ = pd.factorize(df['colorimetrie'])
    encodage_categoriel('transparency')
    encodage_categoriel('teinte')

    # Étape 2 : Ajouter la colonne 'longeur_mm'
    def calculer_longeur_mm(df):
        df['timestamp_first'] = pd.to_datetime(df['timestamp_first'])
        df['timestamp_last'] = pd.to_datetime(df['timestamp_last'])
        df['écart_seccondes'] = (df['timestamp_last'] - df['timestamp_first']).dt.total_seconds()
        vitesse_km_s = 8 / 3600
        df['distance_km'] = df['écart_seccondes'] * vitesse_km_s
        df['longeur_mm'] = (df['distance_km'] * 1000000).round(2)
        df.drop(columns=['distance_km', 'écart_seccondes'], inplace=True)
        return df
    
    df = calculer_longeur_mm(df)

    # Étape 3 : Ajouter la colonne 'height'
    required_height_columns = ['height_40', 'height_70', 'height_110', 'height_200']
    for col in required_height_columns:
        if col not in df.columns:
            df[col] = False  # Ajout de colonnes manquantes avec des valeurs par défaut

    def determine_height(row):
        if not row['height_40'] and not row['height_70'] and not row['height_110'] and not row['height_200']:
            return '[0, 40]'
        if row['height_40'] and not row['height_70'] and not row['height_110'] and not row['height_200']:
            return '[40, 70]'
        elif row['height_40'] and row['height_70'] and not row['height_110'] and not row['height_200']:
            return '[70, 110]'
        elif row['height_40'] and row['height_70'] and row['height_110'] and not row['height_200']:
            return '[110, 200]'
        elif row['height_40'] and row['height_70'] and row['height_110'] and row['height_200']:
            return '>= 200'
        else:
            return 'Undefined'
    
    df['height'] = df.apply(determine_height, axis=1)
    
    height_encoder = {
        '[0, 40]': 1,
        '[40, 70]': 2,
        '[70, 110]': 3,
        '[110, 200]': 4,
        '>= 200': 5
    }
    
    df['height_encoded'] = df['height'].map(height_encoder)
    
    return df


def prediction_tri(dechet: str):
    input_json = json.loads(dechet)
    df = pd.DataFrame([input_json])
    # transformation
    preprocess_data(df)
    
    # Charger le modèle depuis un fichier
    with open('/Users/manu/Desktop/SUP/ML, supervisée non-supervisée/Gestion_dechets/models/random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        
    required_columns = ['metal', 'width', 'colorimetrie_Encoded', 'transparency_Encoded',
                            'teinte_Encoded', 'longeur_mm', 'height_encoded']
    
    # Prédiction
    prediction = loaded_model.predict(df[required_columns])[0]

    return prediction


print(prediction_tri('{"metal":false,"width":86,"optical_code":"T033073","height_40":true,"height_70":true,"height_110":false,"height_200":false,"timestamp_first":"2024-11-14 14:40:59.955602","timestamp_last":"2024-11-14 14:40:59.994822"}'))