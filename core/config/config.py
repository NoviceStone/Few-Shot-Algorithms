# -*- coding: utf-8 -*-
import os
import re
import yaml


def get_cur_path():
    """Get the absolute path of current file.

    Returns: The absolute path of this file (Config.py).

    """
    return os.path.dirname(__file__)


class Config(object):
    """The config parser of `LibFewShot`.

    `Config` is used to parse *.yaml, console params, run_*.py settings to python dict. The rules for resolving merge conflicts are as follows

    1. The merging is recursive, if a key is not be specified, the existing value will be used.
    2. The merge priority is console_params > run_*.py dict > user defined yaml (/LibFewShot/config/*.yaml) > default.yaml (/LibFewShot/core/config/default.yaml)
    """

    def __init__(self, config_file=None, is_resume=False):
        """Initializing the parameter dictionary, actually completes the merging of all parameter definitions.

        Args:
            config_file: Configuration file name. (/LibFewShot/config/name.yaml)
            is_resume: Specifies whether to resume, the default is False.
        """
        self.is_resume = is_resume
        self.file_dict = self._load_config_files(config_file)  # default config from .yaml file
        self.config_dict = self._merge_config_dict()

    def get_config_dict(self):
        """Returns the merged dict.

        Returns:
            dict: A dict of LibFewShot setting.
        """
        return self.config_dict

    @staticmethod
    def _load_config_files(config_file):
        """Parse a YAML file.

        Args:
            config_file (str): Path to yaml file.

        Returns:
            dict: A dict of LibFewShot setting.
        """
        config_dict = dict()
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u"tag:yaml.org,2002:float",
            re.compile(
                u"""^(?:
                     [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list(u"-+0123456789."),
        )

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))
        return config_dict

    def _recur_update(self, dic1, dic2):
        """Merge dictionaries Recursively.

        Used to recursively merge two dictionaries (profiles), `dic2` will overwrite the value of the same key in `dic1`.

        Args:
            dic1 (dict): The dict to be overwritten. (low priority)
            dic2 (dict): The dict to overwrite. (high priority)

        Returns:
            dict: Merged dict.
        """
        if dic1 is None:
            dic1 = dict()
        for k in dic2.keys():
            if isinstance(dic2[k], dict):
                dic1[k] = self._recur_update(dic1[k] if k in dic1.keys() else None, dic2[k])
            else:
                dic1[k] = dic2[k]
        return dic1

    def _update(self, dic1, dic2):
        """Merge dictionaries.

        Used to merge two dictionaries (profiles), `dic2` will overwrite the value of the same key in `dic1`.

        Args:
            dic1 (dict): The dict to be overwritten. (low priority)
            dic2 (dict): The dict to overwrite. (high priority)

        Returns:
            dict: Merged dict.
        """
        if dic1 is None:
            dic1 = dict()
        for k in dic2.keys():
            dic1[k] = dic2[k]
        return dic1

    def _merge_config_dict(self):
        """Merge all dictionaries.

        1. The merging is recursive, if a key is not be specified, the existing value will be used.
        2. The merge priority is console_params > run_*.py dict > user defined yaml (/LibFewShot/config/*.yaml) > default.yaml (/LibFewShot/core/config/default.yaml)

        Returns:
            dict: A LibFewShot setting dict.
        """
        config_dict = dict()
        config_dict = self._update(config_dict, self.file_dict)

        assert config_dict['num_class'] >= 7  # for MSTAR, there are 7 classes at training

        # Modify or add some configs
        config_dict["resume"] = self.is_resume
        if self.is_resume:
            assert config_dict['resume_path'] is None

        return config_dict
