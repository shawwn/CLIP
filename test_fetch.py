# https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
from random import randint
import asyncio
import tqdm
import contextlib
from clip import utils

import httpx
import httpcore
import asyncio
import traceback

from io import BytesIO
import PIL.Image

from argparse import Namespace

from urllib.parse import urlparse
import os
import sys
import random
import posix

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass

sem = asyncio.BoundedSemaphore(64)

async def data_to_url_async(path):
  async with sem:
    data = filebytes(path)
    orig = data
    data = data_to_upload_image_args(data)
    async with httpx.AsyncClient() as client:
      try:
        response = await client.post('https://staging.gather.town/api/uploadImage',
            headers={'Content-Type': 'application/json'},
            data=data,
            timeout=httpx.Timeout(timeout=None))
        if response.status_code == 200:
          url = response.text
          print(json_response(True, url, data=orig, path=path))
          sys.stdout.flush()
        else:
          response.raise_for_status()
      except Exception as caught:
        report_error(caught, data=orig, path=path, code=response.status_code)


async def process(client, callback, url, pbar, fake=False, timeout=None):
  if fake:
    wait_time = randint(1, 2)
    pbar.write('downloading {} will take {} second(s)'.format(url, wait_time))
    await asyncio.sleep(wait_time)  # I/O, context will switch to main function
    pbar.write('downloaded {}'.format(url))
  else:
    noisy = True
    try:
      response = await client.get(url, timeout=httpx.Timeout(timeout=timeout))
      try:
        if response.status_code == 200:
          try:
            await callback(None, response, url=url)
          except:
            with tqdm.tqdm.external_write_mode(file=sys.stdout):
              traceback.print_exc()
        else:
          noisy = False
          response.close()
          await response.raise_for_status()
      finally:
        response.close()
    except Exception as caught:
      if isinstance(caught, (httpx.ConnectError, httpcore.RemoteProtocolError, httpx.RemoteProtocolError)):
        noisy = False
      if noisy:
        with tqdm.tqdm.external_write_mode(file=sys.stdout):
          traceback.print_exc()
      response = Namespace()
      response.url = url
      await callback(caught, response, url=url)
      #report_error(caught, data=orig, path=path, code=response.status_code)
        
def shuffled(items, buffer_size=1_000_000):
  buffer = []
  def pop():
    idx = random.randint(0, len(buffer) - 1)
    return buffer.pop(idx)
  for item in items:
    buffer.append(item)
    if len(buffer) >= buffer_size:
      result = pop()
      yield result
  while len(buffer) > 0:
    result = pop()
    yield result

args = Namespace()
args.args = []
args.concurrency=100
args.maxcount=sys.maxsize

async def main(loop, urls):
  with utils.LineStream() as stream:
    received_bytes = 0
    received_count = 0
    failed_count = 0
    current_item = ''
    dltasks = set()
    def update():
      n = len(dltasks)
      stream.pbar.set_description('%d in-flight / %d done (%d failed) / %.2f MB [%s]' % (n, received_count, failed_count, received_bytes / (1024*1024), current_item))
      stream.pbar.refresh()
    async def callback(err, response, url):
      stream.finish(url)
      nonlocal received_bytes, received_count, failed_count
      if err is not None:
        #stream.pbar.write('Failed: {!r}: {!r}'.format(str(response.url), response))
        failed_count += 1
        update()
        return
      received_count += 1
      received_bytes += len(response.content)
      image_bytes = response.content
      with BytesIO(image_bytes) as bio, PIL.Image.open(bio) as image:
        #url = str(response.url)
        #stream.pbar.write('Received {size} bytes: {url!r} {image!r}'.format(size=len(response.content), url=url, image=image))
        u = urlparse(url)
        #path = u.netloc + u.path
        path = u.scheme + '/' + u.netloc + u.path
        if u.query:
          path += '?' + u.query
        #stream.pbar.write(os.path.join(u.netloc, u.path))
        stream.pbar.write(path)
        update()
    #limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
    async with httpx.AsyncClient(limits=limits) as client:
      for i, url in enumerate(shuffled(stream(urls))):
        if i - args.concurrency >= args.maxcount:
          posix._exit(1)
        current_item = '...' + url.rsplit('/', 1)[-1][-40:]
        if len(dltasks) >= args.concurrency:
          # Wait for some download to finish before adding a new one
          _done, dltasks = await asyncio.wait(dltasks, return_when=asyncio.FIRST_COMPLETED)
        task = process(client, callback, url, pbar=stream.pbar)
        dltasks.add(loop.create_task(task))
      # Wait for the remaining downloads to finish
      await asyncio.wait(dltasks)


if __name__ == '__main__':
  argv = args.args = sys.argv[1:]
  urls = argv[0] if len(argv) >= 1 else 'https://battle.shawwn.com/danbooru2019-s.txt'
  args.concurrency = int(argv[1]) if len(argv) >= 2 else 100
  args.maxcount = int(argv[2]) if len(argv) >= 3 else sys.maxsize
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main(loop, urls))
