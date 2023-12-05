"""
Blip main program
"""
import argparse

from blip.programs.wrapper import parse_command_line_config


def run():
    """
    BLIP main program.
    """
    parser = argparse.ArgumentParser(
        prog='BLIP Module Runner',
        description='This program constructs a BLIP module ' +
                    'from a config file, and then runs the set of modules ' +
                    'in the configuration.',
        epilog='...'
    )
    parser.add_argument(
        'config_file', metavar='<str>.yml', type=str,
        help='config file specification for a BLIP module.'
    )
    parser.add_argument(
        '-n', dest='name', default=None,
        help='name for this run (default None).'
    )
    parser.add_argument(
        '-scratch', dest='local_scratch', default='/local_scratch',
        help='location for the local scratch directory.'
    )
    parser.add_argument(
        '-blip', dest='local_blip', default='/local_blip',
        help='location for the local blip directory.'
    )
    parser.add_argument(
        '-data', dest='local_data', default='/local_data',
        help='location for the local data directory.'
    )
    parser.add_argument(
        '-anomaly', dest='anomaly', default=False,
        help='enable anomaly detection in pytorch'
    )
    args = parser.parse_args()
    meta, module_handler = parse_command_line_config(args)
    module_handler.run_modules()


if __name__ == "__main__":
    run()
