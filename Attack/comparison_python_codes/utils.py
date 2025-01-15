def parse_config(config_file: str) -> dict:
    """Parse configuration file and return dictionary of parameters."""
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=')
                    config[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error reading config file: {e}")
        raise
    return config
