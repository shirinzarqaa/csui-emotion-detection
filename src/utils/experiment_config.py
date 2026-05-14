import os


def get_config(experiment="exp1"):
    is_exp2 = (experiment == "exp2")
    suffix = f"_{experiment}" if is_exp2 else ""

    return {
        "experiment": experiment,
        "mlflow_experiment_p1_trad": f"Traditional_ML_MultiLabel{suffix}",
        "mlflow_experiment_p2_trad": f"Traditional_ML_Final_Test{suffix}",
        "mlflow_experiment_p1_dl": f"Deep_Learning_MultiLabel{suffix}",
        "mlflow_experiment_p2_dl": f"Deep_Learning_Final_Test{suffix}",
        "mlflow_experiment_p1_tf": f"Transformer_MultiLabel{suffix}",
        "mlflow_experiment_p2_tf": f"Transformer_Final_Test{suffix}",
        "checkpoint_dir": "checkpoints",
        "checkpoint_trad": f"checkpoints/traditional_checkpoint{suffix}.json",
        "checkpoint_dl": f"checkpoints/dl_checkpoint{suffix}.json",
        "checkpoint_tf": f"checkpoints/transformers_checkpoint{suffix}.json",
        "analysis_dir": f"analysis{suffix}" if is_exp2 else "analysis",
        "saved_models_dir": f"saved_models{suffix}" if is_exp2 else "saved_models",
        "threshold_tuning": is_exp2,
        "pos_weight": is_exp2,
        "dl_epochs": 30 if is_exp2 else 30,
        "dl_patience": 5 if is_exp2 else 3,
        "dl_max_len": 256 if is_exp2 else 128,
        "tf_epochs": 10 if is_exp2 else 3,
        "tf_patience": 3 if is_exp2 else None,
        "tf_max_len": 256 if is_exp2 else 128,
        "tf_learning_rates": [2e-5, 3e-5, 5e-5] if is_exp2 else [2e-5, 3e-5, 5e-5],
        "tf_batch_sizes": [32],
    }
