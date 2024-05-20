"""
Tools for parsing config files
"""
import yaml

from blip.utils.utils import profiler


class ConfigParser:
    """
    Config file parser for Arrakis.  This class simply
    wraps the pyyaml class with some added functionality,
    such as being able to load nested configs which refer
    to each other.  This can be done by putting

        load_config:    "config_file.yaml"

    in the input config file.  The parameters will be chosen
    in a hierarchical manner (i.e., the parameters can be
    overwritten by respecifying them in higher up configs)
    """

    @profiler
    def __init__(
        self,
        config_file: str,
    ):
        """
        Config file parser for Arrakis initializer

        Args:
            config_file (str): The location of the config
            file to be loaded.
        """
        self.config_file = config_file
        self.nested_config_files = [config_file]
        self.nested_configs = []
        self.data = {}
        self.parse_config()

    @profiler
    def parse_config(self):
        """
        Main function of the config parser.
        """
        self.collect_nested_configs(self.config_file)
        self.nested_config_files.reverse()
        self.nested_configs.reverse()
        for config in self.nested_configs:
            self.data.update(config)

    @profiler
    def collect_nested_configs(
        self,
        config_file: str,
    ):
        """
        Load configs in a hierarchical manner, replacing
        parameters with ones which occur at higher levels.

        Args:
            config_file (_str_): The location of the config
            file to be loaded.
        """
        with open(config_file, 'r') as file:
            temp_data = yaml.safe_load(file)
            self.nested_configs.append(temp_data)
        if 'load_config' in temp_data.keys():
            self.nested_config_files.append(temp_data['load_config'])
            self.collect_nested_configs(temp_data['load_config'])

    @profiler
    def save_config(
        self,
        config_dictionary: str,
        output_file: str,
    ):
        """
        Save a dictionary to a config file.

        Args:
            config_dictionary (_str_): The location of the config
            file to be loaded.
            output_file (_str_): The location and name of the output
            yaml file.
        """
        try:
            with open(output_file, 'w') as file:
                yaml.dump(config_dictionary, file, sort_keys=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save config dictionary {config_dictionary} to output file {output_file}: {e}"
            )
