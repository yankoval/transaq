# -*- coding: windows-1251 -*-
#
# 2024/08 LLM Calculate or show sentiment indicator. Parse listo of urls, calculate indicator and save history to json db
#



from lxml import etree

import requests
from pathlib import Path
from pathvalidate import sanitize_filepath
import json
from datetime import datetime
import argparse
import moex


# logging
import coloredlogs
import logging
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


# get smart-lab ru topic text by url
def getTopic(url:str, xpath=''):
    # Send an HTTP GET request to the given URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page using lxml
        tree = etree.HTML(response.text)

        # Find the element with the specified XPath
        try:
            element_with_content = tree.xpath(xpath)[0]  # '//*[@id="content"]/div[1]/div[1]'
        except Exception as e:
            print(e)
            return ''

        # Extract the text content from the found element
        content_text = element_with_content.xpath('normalize-space()')
        # Joining and stripping leading/trailing whitespace
        return ''.join(content_text).strip()
    else:
        # print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None

def generate(prompt, context, url = 'http://192.168.77.20:11434/api/',model = 'glm4',
                      stream=True):
    url = url + 'generate'
    start_time = datetime.now()
    r = requests.post(url , #'http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                          'context': context,
                          "stream": False,
                          "format": "json",
                      },
                      stream=False,
                      timeout=600)
    r.raise_for_status()
    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        # the response streams one token at a time, print that as we receive it
        # print(f'Response with model:{model}:/n{response_part}')#, end='', flush=False)
#        return response_part
        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            end_time = datetime.now()
            # print('Duration: {}'.format(end_time - start_time))
            return response_part #body['context']
    return r.get('response')


# get top topic id list from smart lab
def getListFromUrl(url='https://www.rbc.ru/quote',
                   xpath='//*[@class="js-load-container"]//div[contains(@class,"js-rm-central-column-item")]//a[@class="q-item__link"]',
                   relativePath=False,  # if true make absolute path from relativ by adding url to result list
                   ):
    # Get popular topik list
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the HTML content of the page using lxml
        tree = etree.HTML(response.text) 
        url = url[:-1] if url[-1] == '/' else url
        res =  [relativePath + el.get('href') for el in tree.xpath(xpath)]
        return res
    return []

def update(db):
    # get sanIndicatorTopicDB
    senIndicatorTopicDB = db.db
    model = db.model
    topTopicListUrl = 'https://smart-lab.ru/'
    questionIndicator  =  """В исходном тексте обзор рынка ценных бумаг, на основе обзора дай рекомендацию покупать или 
    продовать бумаги сейчас. Ответ дай в виде словаря Python где ключ "rec" а значение вещественное число от 0 до 1, 
    где 1 максимально покупать, 0 нейтрально, -1 максимально продовать, если текст не евляется обзором рынка ценных бумаг 
    то None. В ответе не должно быть текста, только число или None."""

    # senIndicatorDbFileName = f'sanIndicatorTopicDB_{model}.json'
    # from pathvalidate import sanitize_filepath
    # senIndicatorDbFileName = sanitize_filepath(senIndicatorDbFileName)
    # # check existing and create if file not exist
    #
    # f = Path(senIndicatorDbFileName)
    # if not f.is_file():
    #     with f.open('w', encoding='utf8') as file:
    #         file.write(json.dumps({'sentimentIndicatorHistory':{}}))
    # read review texts database
    # Calculate sentiment indicator based on smar-lab.ru top topics list parsed by LLM
    # Result saved to json db file in current folder.

    # with f.open('r', encoding='utf8') as file:
    #     senIndicatorTopicDB = json.load(file)
    # senIndicatorTopicDB = db.db


    # get top topic url list
    # topicIdList = getTopTopicList(url=topTopicListUrl)
    # sources list to get topic texts from
    sources = [
        # 'ttps://smart-lab.ru/vtop/
        dict(listFunc = getListFromUrl,
            listKwargs = dict(url='https://smart-lab.ru/vtop/',
                        xpath='//*[@id="content"]//div[contains(@class,"topic")]/h2/a',
                        relativePath='https://smart-lab.ru'
                            ),
              topicFunc = getTopic,
              topicKwargs = dict(xpath='//*[@id="content"]/div[1]/div[1]',)
              ),
        #   https://smart-lab.ru/top/topic/24h/by_comments/
    dict(listFunc = getListFromUrl,
            listKwargs = dict(url='https://smart-lab.ru/top/topic/24h/by_comments/',
                              xpath='//div[@class="topic allbloglist"]/h3/div[@class="inside"]/a',
                              relativePath='https://smart-lab.ru'),
            topicFunc = getTopic,
            topicKwargs = dict(xpath='//*[@id="content"]/div[1]/div[1]',)
        ),
        # https://www.rbc.ru/quote
        dict(listFunc = getListFromUrl,
             listKwargs = dict(url='https://www.rbc.ru/quote',
                               xpath='//*[@class="js-load-container"]//div[contains(@class,"js-rm-central-column-item")]//a[@class="q-item__link"]',
                               relativePath='',),
             topicFunc = getTopic,
             topicKwargs = dict(xpath='(//div[@id="maincontent"]//div[contains(@class,"article__text")])[1]',)
             ),
        ]
    topicIdList = []
    # filter topic already in DB and generate indicator for ather
    with logging_redirect_tqdm():
        for i in tqdm(range(len(sources)), position=0, desc="source", leave=True, colour='green', ncols=80):
            source = sources[i]
            logger.debug('source: '+source['listKwargs']['url'])
            topicUrlList = source['listFunc'](**source['listKwargs'])
            topicIdList += topicUrlList
            toDoList = topicUrlList if forceUpdate else list(set(topicUrlList) - set(senIndicatorTopicDB.keys()))
            logger.info(f'toDoList/topicUrlList: {len(toDoList)}/{len(topicUrlList)}')
            for url_idx in tqdm(range(len(toDoList)), position=1, desc="url_idx", leave=True, colour='red', ncols=80):
                try:
                    topicUrl = toDoList[url_idx]
                    text = source['topicFunc'](topicUrl, **source['topicKwargs'])
                    # logger.info(text)
                    prompt, context = f"""{questionIndicator} Исходный текст: {text}""", []
                    topicResText = generate(prompt, context,model=model)
                    topicRes = json.loads(topicResText)
                    # logger.info(topicRes)
                    senIndicatorTopicDB.update({topicUrl:topicRes})
                    # write updated db
                except Exception as e:
                    logger.error(e)
                    senIndicatorTopicDB.update({topicUrl: {"rec":0.000000001}})
                with db.f.open('w', encoding='utf8') as file:
                    file.write(json.dumps(senIndicatorTopicDB))

    # calculate summary indicator for topic list
    finalRes = {k: v for k, v in senIndicatorTopicDB.items() if k in topicIdList}
    logger.info(finalRes)
    IndicatorList = [v for v in [v.get('rec',0) for k, v in senIndicatorTopicDB.items() if k in topicIdList] if type(v) is float or type(v) is int]
    sentimentIndicator = sum(IndicatorList)/len(IndicatorList)
    sentimentIndicatorHistory = senIndicatorTopicDB.get('sentimentIndicatorHistory',{})
    sentimentIndicatorHistory.update({json.dumps(datetime.now().date().isoformat()): sentimentIndicator})
    senIndicatorTopicDB.update({'sentimentIndicatorHistory': sentimentIndicatorHistory})

    # # write updated db
    # with f.open('w', encoding='utf8') as file:
    #     file.write(json.dumps(senIndicatorTopicDB))
    # logger.info(f'sentimentIndicator: {senIndicatorTopicDB.get("sentimentIndicatorHistory")}')
    # print('\tdate\tsentimentIndicator:\n'+'\n'.join([ f"\t{k:>}\t{senIndicatorTopicDB['sentimentIndicatorHistory'][k]:>8.2f}"
    #         for k in sorted(list(senIndicatorTopicDB['sentimentIndicatorHistory'].keys()), reverse=True)])
    #       )

class DB:
    def __init__(self,  *args, **kwargs):
        self.modelList = [ # model name list
                        "llama3.1:70b",  # 0
                        "glm4:9b-chat-fp16",  # 1
                        "llama3.1:8b",  # 2
                        "llama3.1:70b",  # 3
                        "glm4",  # 4
                        "glm4:9b-chat-q3_K_M",  # 5
                        ]
        if kwargs.get('model') and kwargs.get('model') in self.modelList:
            self.model = kwargs.get('model')
            logger.debug(f'Use model: {self.model}')
        else:
            self.model = self.modelList[0]
            logger.debug(f'Using default model: {self.model}')
    def __enter__(self):
        # read review texts database
        senIndicatorDbFileName = f'sanIndicatorTopicDB_{self.model}.json'
        logger.debug(f'SanIndicatorDbFileName: {senIndicatorDbFileName}')
        # check existing and create if file not exist
        senIndicatorDbFileName = sanitize_filepath(senIndicatorDbFileName)
        self.f = Path(senIndicatorDbFileName)
        if not self.f.is_file():
            logger.info(f'SanIndicatorDbFileName: {senIndicatorDbFileName}, does not exist.')
            with self.f.open('w', encoding='utf8') as file:
                file.write(json.dumps({'sentimentIndicatorHistory':{}}))
            logger.info(f'New db created.')
        with self.f.open('r', encoding='utf8') as file:
            self.db = json.load(file)
        logger.debug(f'db loaded and initialized.')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # write updated db
        with self.f.open('w', encoding='utf8') as file:
            file.write(json.dumps(self.db))

    def show(self,):
        # read and show indicator db
        # show db
        import pandas as pd
        import matplotlib.pyplot as plt
        logger.info(f'sentimentIndicator: {self.db.get("sentimentIndicatorHistory")}')
        d = {k: self.db['sentimentIndicatorHistory'][k] for k in sorted(list(self.db['sentimentIndicatorHistory'].keys()), reverse=False)}
        print('\tdate\tsentimentIndicator:\n' + '\n'.join([f"\t{k:>}\t{self.db['sentimentIndicatorHistory'][k]:>8.2f}"
                                                           for k in sorted(list(d.keys()), reverse=False)]))
        df = pd.DataFrame({'ind':d.values()},
                          index=map(lambda x: datetime.strptime(x[1:-1], '%Y-%m-%d'),d.keys())
                          )
        df.sort_index(inplace=True)
        df.plot(kind='line', title='Sentiment indicator', legend=None)
        plt.hlines(y=0, xmin=df.index.min(), xmax=df.index.max(), colors='red', linestyles='--')
        plt.show()
        pass

if __name__ == "__main__":
    # cProfile.run('main()'#, 'profile_output.txt'
    #              )v
    parser = argparse.ArgumentParser(description='Calculate or show sentiment indicator.')

    # Add arguments for input file, output file, and model
    # parser.add_argument('input_filename', type=str, help='Input file name')
    # parser.add_argument('-o', '--output_filename', dest='output_filename', default='review.txt', type=str, help='Output file name')
    parser.add_argument('-c', '--command', dest='command', type=str, choices=['update', 'show'], default='update',
                        help='"update": calculate and update db. "show": jast read and show db')
    parser.add_argument('-m', '--model', dest='model', type=str, default='llama3.1:70b',
                        help='The model to use (default: "llama3.1:70b")')  # 'llava:13b' 'llama3.1:70b' #'llama3.1:70b' #'glm4:9b-chat-fp16'#'llama3.1:8b', #'llama3.1:70b', 'glm4', #'glm4:9b-chat-q3_K_M'
    parser.add_argument('--verbose', '-v', default='INFO', metavar='VERBOSE',
                        help='Specify log level (ERROR,WARNING,DEBUG,INFO etc. '
                             '[default: INFO]')
    # Parse command line arguments
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    coloredlogs.install(level=logging.ERROR, logger=logger, isatty=True,
                        fmt="%(asctime)s %(levelname)-8s %(message)s",
                        # stream=sys.stdout,
                        datefmt='%Y-%m-%d %H:%M:%S')
    # level = logging.getLevelName(args.verbose)
    logger.setLevel(args.verbose)
    logger.info("Starting...")


    forceUpdate = False
    logger.debug(f"Model: {args.model}, forceUpdate: {forceUpdate}, logging: {args.verbose}.")

    if args.command == 'update':
        logger.info(f'Update...')
        with DB(model=args.model) as db:
            update(db)
    elif args.command == 'show':
        logger.info(f'Show...')
        with DB(model=args.model) as db:
            db.show()
    else:
        logger.error(f"Unknown command: {args.command}")
        exit(-1)