import importlib.metadata

APP_NAME = importlib.metadata.metadata(__package__)["Name"]
