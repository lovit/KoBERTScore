import argparse
from .about import __name__, __version__


def version(args):
    print(f'{__name__}=={__version__}')


def do_task(args):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='PACKAGE Command Line Interface')
    subparsers = parser.add_subparsers(help='PACKAGE SUBPARSER NAME')

    # Show package version
    parser_version = subparsers.add_parser('version', help='Show package version')
    parser_version.set_defaults(func=version)

    # Task parser
    parser_task = subparsers.add_parser('task', help='TASK SUBPARSER')
    parser_task.add_argument('--arg', type=int, help='EXAMPLE')
    parser_task.set_defaults(func=do_task)

    args = parser.parse_args()
    task_function = args.func
    task_function(args)


if __name__ == '__main__':
    main()
