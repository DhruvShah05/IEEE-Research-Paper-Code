from sklearn.ensemble import RandomForestClassifier

def build_model(config: dict) -> RandomForestClassifier:
    """
    Builds a Scikit-Learn RandomForestClassifier based on config.
    """
    model_params = config.get('model_params', {})
    
    n_estimators = model_params.get('n_estimators') or 100
    max_depth = model_params.get('max_depth') or None
    
    # Handle class imbalance explicitly
    imbalance_strat = config.get('imbalance', {}).get('strategy', 'none')
    
    class_weight = "balanced" if imbalance_strat == "class_weight" else None
    
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=config.get('seed', 42),
        n_jobs=-1
    )
