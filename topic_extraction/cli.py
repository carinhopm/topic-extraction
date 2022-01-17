import argparse
import sys

from topic_extraction.scripts import evaluate, predict

COMMANDS = {
    'evaluate': evaluate,
    'predict': predict
}



def main():
    parser = argparse.ArgumentParser(
        description='Topic Extractor',
        usage='topic_extractor <command> [<args>]',
    )
    parser.add_argument('command', type=str, help='Which command to run', choices=COMMANDS.keys())

    args = parser.parse_args(sys.argv[1:2])

    if args.command not in COMMANDS:
        print('Unrecognised command')
        parser.print_help()
        exit(1)

    COMMANDS[args.command]()



if __name__ == '__main__':
    main()
