import pandas as pd
import numpy as np
import json

def sol(obj):
    print(json.dumps(obj, indent=2))

parameters = json.loads(open('algoparams_from_ui.json').read())
target = parameters['design_state_data']['target']
feature_handling = parameters['design_state_data']['feature_handling']

df = pd.read_csv("iris.csv")

# data pre-processing
for col, feature in feature_handling.items():
    if not feature['is_selected']:
        df.drop(col, axis=1, inplace=True)

    if feature["feature_variable_type"] == "numerical":

        if feature['feature_details']["missing_values"] == "Impute":
            if feature['feature_details']['impute_with'] == "Average of values":
                df[col].fillna(df[col].mean(), inplace=True)
            elif feature['feature_details']['impute_with'] == "custom":
                df[col].fillna(feature['feature_details']
                               ['impute_value'], inplace=True)
                
    elif feature["feature_variable_type"] == "text":
        labels = {key: num for num, key in enumerate(df[col].unique())}
        df[col] = df[col].apply(lambda x: labels[x])

# feature reduction
config = parameters['design_state_data']['feature_reduction']

target_col = target['target']

X = df.drop(target_col, axis=1).values
y = df[target_col].values

if config['feature_reduction_method'] == "Tree-based":
    if target['type'] == "regression":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        sel = SelectFromModel(RandomForestRegressor(n_estimators=int(
            config['num_of_trees']), max_depth=int(config['depth_of_trees'])))

    elif target['type'] == "classification":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        sel = SelectFromModel(RandomForestClassifier(n_estimators=int(
            config['num_of_trees']), max_depth=int(config['depth_of_trees'])))

    sel.fit(X, y)
    feature_importance = sel.estimator_.feature_importances_
    sorted_indices = np.argsort(feature_importance)[::-1]
    keep_columns = df.columns[np.concatenate((sorted_indices[:int(
        config['num_of_features_to_keep'])], [list(df.columns).index(target_col)]))]
    df = df[keep_columns]

elif config['feature_reduction_method'] == "No Reduction":
    pass

elif config['feature_reduction_method'] == "Correlation with target":
    corr = df.corr()[target_col].drop(target_col)
    sorted_cor = sorted(dict(abs(corr).items()).items(), key=lambda x: x[1], reverse=True)[
        :int(config['num_of_features_to_keep'])]
    keep_columns = np.array([key for key, value in sorted_cor] + [target_col])
    df = df[keep_columns]

elif config['feature_reduction_method'] == "Principal Component Analysis":
    from sklearn.decomposition import PCA

    pca = PCA(n_components=int(config['num_of_features_to_keep']))
    pca.fit(X)
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    X = pca.transform(X)
    

algorithms = parameters['design_state_data']['algorithms']


def best_model(algo_name, hyperparameters):
    model = None
    model_name = hyperparameters.pop('model_name')

    if model_name == "Random Forest Regressor":
        from sklearn.ensemble import RandomForestRegressor

        from sklearn.model_selection import GridSearchCV
        parameters = {
            'n_estimators': [hyperparameters["min_trees"], hyperparameters["max_trees"]],
            'max_depth': [hyperparameters["min_depth"], hyperparameters["max_depth"]],
            'min_samples_leaf': [hyperparameters["min_samples_per_leaf_min_value"],
                                 hyperparameters["min_samples_per_leaf_max_value"]]
            }
        model = GridSearchCV(RandomForestRegressor(), parameters, cv=5, n_jobs=-1)
        model.fit(X, y)
        print(f"Best parameters: {model.best_params_}")
        print(f"Best score: {model.best_score_}")
        return model.best_estimator_

    # model selection
    elif model_name == "Gradient Boosted Trees":
        pass
    elif model_name == "Gradient Boosted Trees":
        pass
    elif model_name == "LinearRegression":
        pass
    elif model_name == "LogisticRegression":
        pass
    elif model_name == "RidgeRegression":
        pass
    elif model_name == "Lasso Regression":
        pass
    elif model_name == "Lasso Regression":
        pass
    elif model_name == "XG Boost":
        pass
    elif model_name == "Decision Tree":
        pass
    elif model_name == "Decision Tree":
        pass
    elif model_name == "Support Vector Machine":
        pass
    elif model_name == "Stochastic Gradient Descent":
        pass
    elif model_name == "KNN":
        pass
    elif model_name == "Extra Random Trees":
        pass
    elif model_name == "Neural Network":
        pass
    
    return model

for algo_name, hyperparameters in algorithms.items():
    is_selected = hyperparameters.pop('is_selected')
    name = hyperparameters['model_name']

    if not is_selected:
        continue

    model = best_model(algo_name, hyperparameters)
    sol(name)
    sol(hyperparameters)
    
    if not model is None:
        break