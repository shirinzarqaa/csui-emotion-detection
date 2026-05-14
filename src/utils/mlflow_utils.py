import time
import mlflow
from loguru import logger

_MAX_RETRIES = 5
_RETRY_DELAY = 10


def _retry(fn, *args, **kwargs):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_DELAY * attempt
                logger.warning(f"MLflow call failed (attempt {attempt}/{_MAX_RETRIES}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"MLflow call failed after {_MAX_RETRIES} attempts, skipping: {e}")
                return None


def safe_set_experiment(name):
    return _retry(mlflow.set_experiment, name)


def safe_start_run(run_name=None, experiment_id=None):
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            kwargs = {}
            if run_name:
                kwargs["run_name"] = run_name
            if experiment_id:
                kwargs["experiment_id"] = experiment_id
            return mlflow.start_run(**kwargs)
        except Exception as e:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_DELAY * attempt
                logger.warning(f"mlflow.start_run failed (attempt {attempt}/{_MAX_RETRIES}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"mlflow.start_run failed after {_MAX_RETRIES} attempts — running WITHOUT MLflow tracking")
                return None


def safe_end_run():
    try:
        mlflow.end_run()
    except Exception as e:
        logger.warning(f"mlflow.end_run failed: {e}")


def safe_log_param(key, value):
    _retry(mlflow.log_param, key, value)


def safe_log_params(params_dict):
    for k, v in params_dict.items():
        safe_log_param(k, v)


def safe_log_metric(key, value, step=None):
    _retry(mlflow.log_metric, key, value, step=step)


def safe_log_metrics(metrics_dict, step=None):
    for k, v in metrics_dict.items():
        safe_log_metric(k, v, step=step)


def safe_set_tag(key, value):
    _retry(mlflow.set_tag, key, value)
