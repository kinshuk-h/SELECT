"""

    network
    ~~~~~~~

    Provides utility functions for network-related tasks, such as downloading files from Google Drive
        or executing SPARQL queries on an online knowledge base

"""

import sys

import bs4
import pandas
import requests
import SPARQLWrapper
import tqdm.auto as tqdm

wikidata_engine = SPARQLWrapper.SPARQLWrapper("https://query.wikidata.org/sparql")
yago_engine     = SPARQLWrapper.SPARQLWrapper("https://yago-knowledge.org/sparql/query")

def run_wikidata_query(query):
    wikidata_engine.setReturnFormat('json')
    wikidata_engine.setQuery(query)
    results = wikidata_engine.query().convert()
    return pandas.json_normalize(results['results']['bindings'])

def run_yago_query(query):
    yago_engine.setReturnFormat('json')
    yago_engine.setQuery(query)
    results = yago_engine.query().convert()
    return pandas.json_normalize(results['results']['bindings'])

def download_from_drive(drive_link, target_path, bufsize=4096, debug=False):
    """ Downloads a file from a Google Drive Link.

    Args:
        drive_link (str): Download link to the file: contains export=download
        target_path (str): Target location to download to.
        bufsize (int, optional): Size of chunks to download per iteration. Defaults to 4096.
        debug (bool, optional): Dump debugging info to STDERR. Defaults to False.
    """
    response = requests.get(drive_link, stream=True)
    response.raise_for_status()
    if 'html' in response.headers['Content-Type']:
        response = requests.get(drive_link)
        response.raise_for_status()
        page = bs4.BeautifulSoup(response.text, features="lxml")
        if form := page.find('form', id='download-form'):
            id   = form.select_one("input[name='id']")['value']
            uuid = form.select_one("input[name='uuid']")['value']
            data = { 'confirm': 't', 'export': 'download', 'id': id, 'uuid': uuid }
            response = requests.get(page.find('form')['action'], params=data, stream=True)
            response.raise_for_status()

    with open(target_path, 'wb+') as file:
        with tqdm.tqdm(
            total=int(response.headers['Content-Length']),
            unit='B', unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=bufsize):
                file.write(chunk)
                pbar.update(len(chunk))

    if debug:
        print("Downloaded to", target_path, file=sys.stderr)