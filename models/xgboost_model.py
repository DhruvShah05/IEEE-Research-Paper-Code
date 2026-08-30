import xgboost as xgb

def build_model(config: dict) -> xgb.XGBClassifier:
    """
    Builds an XGBoost Classifier based on config.
    Note: XGBoost handles class weights via sample_weight during .fit(), 
    not in the constructor for multi-class classification natively in standard scikit-learn API,
    or via `scale_pos_weight` for binary. For multi-class, we must compute sample_weights in the training loop.
    Here we just initialize the architecture.
    """
    model_params = config.get('model_params', {})
    
    n_estimators = model_params.get('n_estimators') or 100
    max_depth = model_params.get('max_depth') or 6
    
    # We use tree_method='hist' for speed on large datasets
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        objective='multi:softprob',
        num_class=3,
        random_state=config.get('seed', 42),
        n_jobs=-1,
        tree_method='hist'
    )
