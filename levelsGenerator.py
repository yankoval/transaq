#       24/12
#       Generate levels for given tickers and write to file for transaq ATF script
#
import os
import coloredlogs
import logging
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
import constants
from pathlib import Path
import textwrap


def main(input_filename, output_filename, model, logger):
    logger.debug('main')


if __name__ == "__main__":
    # cProfile.run('main()'#, 'profile_output.txt'
    #              )
    parser = argparse.ArgumentParser(description='Process some parameters.',
                                     epilog=textwrap.dedent('''   additional information:
             If you vont to use .ini file put __CONSTANTS__=DEFAULT env variable 
             and create programm_name.ini file with content: 
             [DEFAULT] 
             something = a_default_value
             [a_section]
             something = a_section_value
         '''))

    # Add arguments for input file, output file, and model
    parser.add_argument('-input_filename', type=str, dest='input_filename', default='input_filename.txt',
                        help='Input file name')
    parser.add_argument('-o', '--output_filename', dest='output_filename', default='output_filename.txt', type=str,
                        help='Output file name')
    parser.add_argument('-m', '--model', dest='model', type=str, default='llama3.1:8b',
                        help='The model to use (default: "llama3.1:8b")')  # 'llama3.1:70b' #'llama3.1:70b' #'glm4:9b-chat-fp16'#'llama3.1:8b', #'llama3.1:70b', 'glm4', #'glm4:9b-chat-q3_K_M'

    # Parse command line arguments
    args = parser.parse_args()

    # logging
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=logging.DEBUG, logger=logger, isatty=True,
                        fmt="%(asctime)s %(levelname)-8s %(message)s",
                        stream=sys.stderr,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger.debug("Procesing:" + args.input_filename)

    # constants `section:
    # if you want to use ini file put __CONSTANTS__=DEFAULT enviroment varible
    # use ini file with programm file name
    # [DEFAULT]
    # something = a_default_value
    # all =  1
    # a_string = 0350
    #
    # [a_section]
    # something = a_section_value
    # just_for_m`e = 5.0
    if '__CONSTANTS__' in os.environ:
        consts = constants.Constants(  # variable='__CONSTANTS__',
            filename=Path(__file__).with_suffix('.ini'))  # doctest: +SKIP
        logger.debug(consts)
    # Call the main function with the parsed arguments
    main(args.input_filename, args.output_filename, args.model, logger)
