import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--debug',
        action='store_true',
        dest='debug',
        default='False',
        help='Turns debugging on')
    args = argparser.parse_args()
    return args
