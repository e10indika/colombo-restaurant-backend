# Training package — split into focused modules:
#   spark_session.py    — SparkSession factory for training
#   data_loader.py      — load restaurant CSV + assign integer IDs
#   rating_generator.py — synthetic user-rating matrix generation
#   model_trainer.py    — ALS fit + RMSE evaluation
#   artefact_builder.py — build and persist top_restaurants.json
