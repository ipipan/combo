import errno
import logging
import math
import os

import requests
import tqdm
import urllib3
from requests import adapters, exceptions

logger = logging.getLogger(__name__)

_URL = "http://mozart.ipipan.waw.pl/~mklimaszewski/models/{name}.tar.gz"
_HOME_DIR = os.getenv("HOME", os.curdir)
_CACHE_DIR = os.getenv("COMBO_DIR", os.path.join(_HOME_DIR, ".combo"))


def download_file(model_name, force=False):
    _make_cache_dir()
    url = _URL.format(name=model_name)
    local_filename = url.split("/")[-1]
    location = os.path.join(_CACHE_DIR, local_filename)
    if os.path.exists(location) and not force:
        logger.debug("Using cached model.")
        return location
    chunk_size = 1024 * 10
    logger.info(url)
    try:
        with _requests_retry_session(retries=2).get(url, stream=True) as r:
            total_length = math.ceil(int(r.headers.get("content-length")) / chunk_size)
            with open(location, "wb") as f:
                with tqdm.tqdm(total=total_length) as pbar:
                    for chunk in r.raw.stream(chunk_size, decode_content=False):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            pbar.update(1)
    except exceptions.RetryError:
        raise ConnectionError(f"Couldn't find or download model {model_name}.tar.gz. "
                              "Check if model name is correct or try again later!")

    return location


def _make_cache_dir():
    try:
        os.makedirs(_CACHE_DIR)
        logger.info(f"Making cache dir {_CACHE_DIR}")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(404, 500, 502, 504),
    session=None,
):
    """Source: https://www.peterbe.com/plog/best-practice-with-retries-with-requests"""
    session = session or requests.Session()
    retry = urllib3.Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = adapters.HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
