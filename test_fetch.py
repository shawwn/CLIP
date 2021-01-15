# https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
from random import randint
import asyncio
import tqdm
import contextlib
from clip import utils

import httpx
import asyncio
import traceback

from io import BytesIO
import PIL.Image

from argparse import Namespace

from urllib.parse import urlparse
import os

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


async def process(callback, url, pbar, fake=False, timeout=60.0):
  if fake:
    wait_time = randint(1, 2)
    pbar.write('downloading {} will take {} second(s)'.format(url, wait_time))
    await asyncio.sleep(wait_time)  # I/O, context will switch to main function
    pbar.write('downloaded {}'.format(url))
  else:
    async with httpx.AsyncClient() as client:
      try:
        response = await client.get(url, timeout=httpx.Timeout(timeout=timeout))
        if response.status_code == 200:
          try:
            await callback(None, response)
          except:
            with tqdm.tqdm.external_write_mode(file=sys.stdout):
              traceback.print_exc()
        else:
          await response.raise_for_status()
      except Exception as caught:
        #traceback.print_exc()
        response = Namespace()
        response.url = url
        await callback(caught, response)
        #report_error(caught, data=orig, path=path, code=response.status_code)
        

async def main(loop, url, concurrency=100):
  with utils.LineStream() as stream:
    received_bytes = 0
    received_count = 0
    failed_count = 0
    dltasks = set()
    async def callback(err, response):
      nonlocal received_bytes, received_count, failed_count
      if err is not None:
        #stream.pbar.write('Failed: {!r}: {!r}'.format(str(response.url), response))
        failed_count += 1
        return
      received_count += 1
      received_bytes += len(response.content)
      image_bytes = response.content
      image = PIL.Image.open(BytesIO(image_bytes))
      url = str(response.url)
      #stream.pbar.write('Received {size} bytes: {url!r} {image!r}'.format(size=len(response.content), url=url, image=image))
      u = urlparse(url)
      path = u.netloc + u.path
      #stream.pbar.write(os.path.join(u.netloc, u.path))
      stream.pbar.write(path)
    for i, line in enumerate(stream(url)):
        n = len(dltasks)
        stream.pbar.set_description('%d in-flight / %d finished (%d failed) / %.2f MB [%s]' % (n, received_count, failed_count, received_bytes / (1024*1024), line.rsplit('/', 1)[-1]))
        if len(dltasks) >= concurrency:
            # Wait for some download to finish before adding a new one
            _done, dltasks = await asyncio.wait(
                dltasks, return_when=asyncio.FIRST_COMPLETED)
        task = process(callback, line, pbar=stream.pbar)
        dltasks.add(loop.create_task(task))
    # Wait for the remaining downloads to finish
    await asyncio.wait(dltasks)


if __name__ == '__main__':
  import sys
  args = sys.argv[1:]
  url = args[0] if len(args) >= 1 else 'https://battle.shawwn.com/danbooru2019-s.txt'
  concurrency = int(args[1]) if len(args) >= 2 else 100
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main(loop, url, concurrency=concurrency))
