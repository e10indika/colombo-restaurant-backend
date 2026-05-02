"""
training/model_trainer.py — ALS model training and RMSE evaluation.
"""

import logging
import os

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def train_als(
    ratings_df: DataFrame,
    train_split: list,
    seed: int,
    rank: int,
    max_iter: int,
    reg_param: float,
) -> tuple[ALSModel, float]:
    """
    Split ratings into train/test, fit an ALS model, and return the model
    together with the test RMSE.

    Args:
        ratings_df:  DataFrame with columns userId, restaurantId, rating.
        train_split: [train_fraction, test_fraction], e.g. [0.8, 0.2].
        seed:        Random seed for the split.
        rank:        Number of ALS latent factors.
        max_iter:    Maximum ALS iterations.
        reg_param:   L2 regularisation parameter.

    Returns:
        (model, rmse) tuple.
    """
    train_df, test_df = ratings_df.randomSplit(train_split, seed=seed)

    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol='userId',
        itemCol='restaurantId',
        ratingCol='rating',
        coldStartStrategy='drop',
        nonnegative=True,
    )

    logger.info('Training ALS (rank=%d, maxIter=%d, regParam=%s) …', rank, max_iter, reg_param)
    model = als.fit(train_df)

    predictions = model.transform(test_df)
    evaluator   = RegressionEvaluator(
        metricName='rmse',
        labelCol='rating',
        predictionCol='prediction',
    )
    rmse = evaluator.evaluate(predictions)
    logger.info('Test RMSE: %.4f', rmse)
    return model, rmse


def save_model(model: ALSModel, output_dir: str) -> None:
    """Persist the trained ALSModel to disk, overwriting any previous version."""
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    model.write().overwrite().save(output_dir)
    logger.info('ALS model saved to %s', output_dir)
