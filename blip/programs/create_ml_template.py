"""
Script for generating an ML template with an accompanying config file
"""
import os
import shutil
import argparse

from blip.utils.utils import get_datetime


def run():
    parser = argparse.ArgumentParser(
        prog='BLIP ML Module Creator',
        description='This program constructs a BLIP ML Module ' +
                    'with a template for a custom model/loss/metric/callback and dataset.' +
                    '  If also provides an empty config file.',
        epilog='...'
    )
    parser.add_argument(
        '-module_name', dest='module_name', default='my',
        help='name to use for the custom module parts.'
    )
    parser.add_argument(
        '-module_location', dest='module_location', default='/local_scratch',
        help='location for the local scratch directory.'
    )

    args = parser.parse_args()
    if args.module_location is not None:
        if not os.path.isdir(args.module_location):
            args.module_location = './'
    else:
        args.module_location = './'
    snake_case_name = args.module_name
    module_name_words = snake_case_name.split('_')
    camel_case_name = ''.join(word.capitalize() for word in module_name_words)
    module_location = args.module_location + '/' + camel_case_name + 'CustomModule/'

    # now copy template files and replace 'PlaceHolder' with args.module_name
    if not os.path.isdir(module_location):
        os.makedirs(module_location)
    else:
        now = get_datetime()
        if not os.path.isdir(f"{module_location}.backup/"):
            os.makedirs(f"{module_location}.backup/")
        os.makedirs(f"{module_location}.backup/{now}")
        selected_files = [
            file for file in os.listdir(module_location)
            if (file.endswith('.py') or file.endswith('.yaml'))
        ]
        for file in selected_files:
            source_path = os.path.join(module_location, file)
            destination_path = os.path.join(f"{module_location}.backup/{now}", file)
            shutil.move(source_path, destination_path)

    for module_file in ['callback', 'dataset', 'loss', 'metric', 'model']:
        shutil.copy(
            os.path.dirname(__file__) + f'/blank_ml_template/blank_{module_file}.py',
            f'{module_location}/{snake_case_name}_{module_file}.py'
        )
        with open(f'{module_location}/{snake_case_name}_{module_file}.py', 'r') as file:
            content = file.read()
        modified_content = content.replace(
            'PlaceHolder',
            camel_case_name
        )
        modified_content = modified_content.replace(
            'place_holder',
            snake_case_name
        )
        with open(f'{module_location}{snake_case_name}_{module_file}.py', 'w') as file:
            file.write(modified_content)

    # same for config file
    shutil.copy(
        os.path.dirname(__file__) + '/blank_ml_template/blank_config.yaml',
        f'{module_location}/{snake_case_name}_config.yaml'
    )
    with open(f'{module_location}/{snake_case_name}_config.yaml', 'r') as file:
        content = file.read()
    modified_content = content.replace(
        'PlaceHolder',
        camel_case_name
    )
    modified_content = modified_content.replace(
        'place_holder',
        snake_case_name
    )
    with open(f'{module_location}{snake_case_name}_config.yaml', 'w') as file:
        file.write(modified_content)


if __name__ == "__main__":
    run()
