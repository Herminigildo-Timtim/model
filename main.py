import torch
import pandas as pd
import numpy as np
import scipy.sparse
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from informer_model import Informer
from mtcn_model import train_and_predict, TimeSeriesDataset


def preprocess_data_feature(file_path, target_column):
    df = pd.read_csv(file_path)

    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns found: {df.columns[df.columns.duplicated()]}")

    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df.dropna(inplace=True)

    target_series = df[target_column]
    feature_data = df.drop(columns=[target_column])

    start_date = df['Date Time'].min()
    df['datetime_feature'] = (df['Date Time'] - start_date).dt.total_seconds() / 3600
    datetime_feature = df['datetime_feature'].values.reshape(-1, 1)

    feature_data = feature_data.drop(columns=['Date Time'])

    numerical_cols = feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns.tolist()

    transformers = [('num', MinMaxScaler(), numerical_cols)]

    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))

    preprocessor = ColumnTransformer(transformers)
    transformed_features = preprocessor.fit_transform(feature_data)

    if scipy.sparse.issparse(transformed_features):
        transformed_features = transformed_features.toarray()

    features_with_datetime = np.hstack((transformed_features, datetime_feature))

    feature_names = numerical_cols
    if categorical_cols:
        try:
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        except (KeyError, AttributeError):
            pass
    feature_names.append('datetime_feature')

    scalers = {
        'numeric': preprocessor.named_transformers_['num'],
        'feature_count': features_with_datetime.shape[1]
    }

    return {
        'features': features_with_datetime,
        'feature_names': feature_names,
        'date': df['Date Time'],
        'target': target_series,
        'scalers': scalers  
    }


def perform_feature_selection(file_path, selection_method, selection_threshold):
    data_config = preprocess_data_feature(file_path, target_column='T (degC)')

    features = data_config['features']
    feature_names = data_config['feature_names']

    if 'scalers' not in data_config:
        raise KeyError("Error: 'scalers' not found in preprocess_data_feature() output!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(features, dtype=torch.float32).to(device)

    model = Informer(
        enc_in=x.shape[1],
        selection_method=selection_method,
        selection_threshold=selection_threshold
    ).to(device)

    with torch.no_grad():
        importance_scores = model.compute_feature_importance(x)

    max_importance = importance_scores.max().item()
    selected_indices = [i for i, score in enumerate(importance_scores) if score.item() == max_importance]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    selected_feature_data = features[:, selected_indices]

    final_df = pd.concat([
        data_config['date'],
        pd.DataFrame(selected_feature_data, columns=selected_feature_names),
        data_config['target']
    ], axis=1)

    return {
        'selected_features': selected_feature_names,
        'full_dataframe': final_df,
        'date': data_config['date'],
        'target': data_config['target'],
        'scalers': data_config['scalers']
    }


def time_temporal_features_extraction(training_df):
    training_df['day_of_week'] = training_df['Date Time'].dt.dayofweek
    training_df['week'] = training_df['Date Time'].dt.isocalendar().week
    training_df['month'] = training_df['Date Time'].dt.month
    return training_df


def collate_fn(batch):
    inputs = [item['inputs'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_length = max(label.size(0) for label in labels)
    labels_padded = [torch.nn.functional.pad(label, (0, max_length - label.size(0))) for label in labels]

    inputs_padded = torch.stack(inputs)
    labels_padded = torch.stack(labels_padded)

    return {'inputs': inputs_padded, 'labels': labels_padded}


def main():
    file_path = "jena_climate_2009_2016.csv/jena_climate_2009_2016.csv"

    selection_results = perform_feature_selection(file_path, 'importance', 0.5)
    training_df = selection_results['full_dataframe']
    training_df.ffill(inplace=True)

    print("Selected Features:", selection_results['selected_features'])

    training_df.dropna(subset=['Date Time'], inplace=True)
    training_df.drop_duplicates(subset=['Date Time'], inplace=True)

    training_df = time_temporal_features_extraction(training_df)

    selected_feature_names = selection_results['selected_features'] + ['day_of_week', 'week', 'month']

    features = training_df[selected_feature_names].values
    targets = training_df['T (degC)'].values

    features = features.astype(np.float32)

    features_tensor = torch.FloatTensor(features)
    targets_tensor = torch.FloatTensor(targets)

    sequence_length = 30
    num_epochs = 50
    batch_size = 250

    trainer, history = train_and_predict(
        features=features_tensor,
        targets=targets_tensor,
        sequence_length=sequence_length,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    torch.save(trainer.model.state_dict(), 'trained_model.pth')

    joblib.dump(selection_results['scalers'], 'scalers.pkl')

    print("Model and scalers saved successfully!")

    test_dataset = TimeSeriesDataset(features, targets, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = trainer.model
    model.eval()
    predictions = []
    actual_values = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(trainer.device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy().flatten())
            actual_values.append(batch_y.numpy().flatten())

    predictions = np.concatenate(predictions)
    actual_values = np.concatenate(actual_values)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual')
    plt.plot(predictions, label='Predicted', linestyle='--')
    plt.title('Actual vs Predicted T (degC)')
    plt.xlabel('Time Steps')
    plt.ylabel('T (degC)')
    plt.legend()
    plt.show()

    print("\nSample Predictions:")
    for i in range(10):
        print(f"Actual: {actual_values[i]:.4f}, Predicted: {predictions[i]:.4f}")


if __name__ == "__main__":
    main()
