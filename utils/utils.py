import importlib.util


def load_config(config_abs_path):
	spec = importlib.util.spec_from_file_location("config", config_abs_path)
	config_module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(config_module)
	return config_module.config


