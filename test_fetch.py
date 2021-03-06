# https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
from random import randint
import asyncio
import tqdm
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

import tempfile

from functools import partial
import aiofile
from clip import contextlib
import time

def maketree(path):
  try:
    os.makedirs(path)
  except:
    pass

def atomicwrite(path, mode='wb', overwrite=True):
  #root='/Volumes/birdie/'; path=root+'http/c8.alamy.com/comp/K07NFE/russian-soldiers-on-parade-in-a-street-during-the-first-world-war-K07NFE.jpg'; writer = atomicwrites.AtomicWriter(path, mode='wb'); ctx = writer._open(partial(writer.get_fileobject, dir=root+'tmp')); f = ctx.__enter__()
  dir = os.path.join(args.root, 'tmp')
  maketree(dir)
  parentdir = os.path.dirname(path)
  if parentdir:
    maketree(parentdir)
  import atomicwrites
  writer = atomicwrites.AtomicWriter(path, mode=mode, overwrite=overwrite)
  ctx = writer._open(partial(writer.get_fileobject, dir=dir))
  return ctx


def toomany(opener, *args, delay=1.0, **kws):
  attempts = 0
  while True:
    attempts += 1
    try:
      return opener(*args, **kws)
    except OSError as e:
      if e.errno == 24 and attempts < 500:
        time.sleep(delay)
      else:
        raise


@contextlib.asynccontextmanager
async def atoomany(opener, *args, delay=1.0, **kws):
  attempts = 0
  while True:
    attempts += 1
    try:
      async with opener(*args, **kws) as f:
        yield f
        return
    except OSError as e:
      if e.errno == 24 and attempts < 500:
        await asyncio.sleep(delay)
      else:
        raise

@contextlib.contextmanager
def atomicwrite2(path, mode='wb', overwrite=True):
  #root='/Volumes/birdie/'; path=root+'http/c8.alamy.com/comp/K07NFE/russian-soldiers-on-parade-in-a-street-during-the-first-world-war-K07NFE.jpg'; writer = atomicwrites.AtomicWriter(path, mode='wb'); ctx = writer._open(partial(writer.get_fileobject, dir=root+'tmp')); f = ctx.__enter__()
  dir = os.path.join(args.root, 'tmp')
  maketree(dir)
  parentdir = os.path.dirname(path)
  if parentdir:
    maketree(parentdir)
  descriptor, name = toomany(tempfile.mkstemp, suffix='.tmp', dir=dir)
  os.close(descriptor)
  try:
    with toomany(open, name, mode) as f:
      yield f
    os.rename(src=name, dst=path)
  except:
    try:
      os.unlink(name)
    except:
      pass
    raise




def writebytes(path, filebytes, **kws):
  with atomicwrite2(path, **kws) as f:
    f.write(filebytes)

import concurrent.futures

async def awritebytes(path, filebytes, **kws):
  loop = asyncio._get_running_loop()
  pool = None # concurrent.futures.ThreadPoolExecutor()
  await loop.run_in_executor(pool, lambda: writebytes(path, filebytes, **kws))

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


async def process(client, callback, url, stream, fake=False, timeout=30.0):
  pbar = stream.pbar
  if fake:
    wait_time = randint(1, 2)
    pbar.write('downloading {} will take {} second(s)'.format(url, wait_time))
    await asyncio.sleep(wait_time)  # I/O, context will switch to main function
    pbar.write('downloaded {}'.format(url))
  else:
    noisy = True
    try:
      filepath = url2path(url)
      if os.path.isfile(filepath):
        if args.skip_downloaded:
          noisy = False
          raise ValueError('Skipped')
        else:
          response = Namespace()
          response.url = url
          response.close = lambda: None
          response.status_code = 200
          async with atoomany(aiofile.async_open, filepath, 'rb') as f:
            response.content = await f.read()
          await callback(None, response, url=url)
      else:
        response = await client.get(url, timeout=httpx.Timeout(timeout=timeout))
        try:
          if response.status_code == 200:
            try:
              #writebytes(filepath, response.content)
              await awritebytes(filepath, response.content)
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
    finally:
      stream.finish(url)
        
def shuffled(items, start=0):
  buffer = []
  def pop():
    idx = random.randint(0, len(buffer) - 1)
    return buffer.pop(idx)
  for i, item in enumerate(items):
    if i < start:
      continue
    buffer.append(item)
    if args.shufflesize >= 0 and len(buffer) >= args.shufflesize:
      result = pop()
      yield result
  while len(buffer) > 0:
    result = pop()
    yield result

args = Namespace()
args.args = []
args.concurrency=50
args.skip_downloaded = True
args.shufflesize = -1
args.start=0
args.maxcount=sys.maxsize
args.root = os.path.join(os.getcwd(), 'download')

def url2path(url, root=None):
  u = urlparse(url)
  #path = u.netloc + u.path
  path = u.scheme + '/' + u.netloc + u.path
  if u.query:
    path += '?' + u.query
  filepath = os.path.join(root or args.root, path)
  return filepath

import mock
import threading
from types import SimpleNamespace as NS
from collections import OrderedDict
import functools

mocks = globals().get('mocks') or NS(advice=OrderedDict({}), deactivate=None, lock=threading.RLock())

def mocks_active():
  return mocks.deactivate is not None

def mock_function(unique_name, lib, name=None, doc=None):
  return mock_method(unique_name, lib, name=name, doc=doc, toplevel=True)

def mock_method(unique_name, cls, name=None, doc=None, toplevel=False):
  def func(fn):
    nonlocal name
    if name is None:
      name = fn.__name__
    wrapped = getattr(cls, name)
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
      return fn(wrapped, cls, *args, **kwargs)
    if hasattr(wrapped, '__name__'):
      _fn.__name__ = wrapped.__name__
    if hasattr(wrapped, '__module__'):
      _fn.__module__ = wrapped.__module__
    if hasattr(wrapped, '__qualname__'):
      _fn.__qualname__ = wrapped.__qualname__
    if toplevel:
      mocks.advice[unique_name] = lambda: mock.patch.object(cls, name, side_effect=_fn)
    else:
      mocks.advice[unique_name] = lambda: mock.patch.object(cls, name, _fn)
    return _fn
  return func

def deactivate_mocks():
  with mocks.lock:
    if mocks.deactivate:
      mocks.deactivate()
      mocks.deactivate = None
      return True

def activate_mocks():
  with mocks.lock:
    deactivate_mocks()
    with contextlib.ExitStack() as stack:
      for creator in mocks.advice.values():
        stack.enter_context(creator())
      stk = stack.pop_all()
      mocks.deactivate = stk.close
      return stk


@mock_method('patch_set_cookie_header', httpx._models.Cookies, 'set_cookie_header',
    doc="""Disable cookies for performance""")
def patch_set_cookie_header(*args, **kws):
  #import pdb; pdb.set_trace()
  #print('patched')
  #sys.stdout.flush()
  pass

import http.cookiejar

@mock_method('patch_set_cookie', http.cookiejar.CookieJar, 'set_cookie',
    doc="""Disable cookies for performance""")
def patch_set_cookie(*args, **kws):
  #import pdb; pdb.set_trace()
  #print('patched')
  #sys.stdout.flush()
  pass

async def main(loop, urls):
  with utils.LineStream() as stream:
    received_bytes = 0
    received_count = 0
    failed_count = 0
    current_item = ''
    dltasks = set()
    def update():
      if (received_count + failed_count) % 50 == 0:
        n = len(dltasks)
        t = stream.pbar.n / stream.pbar.total
        est_bytes = int(received_bytes / t) if t > 0 else 0
        est_count = int((received_count + failed_count) / t) if t > 0 else 0
        item = '...' + current_item.rsplit('/', 1)[-1][-40:]
        stream.pbar.set_description('%d in-flight | %d done + %d failed of ~%s | %.2f MB of ~%.2f GB [%s]' % (
          n, received_count, failed_count, '{:,}'.format(est_count), received_bytes / (1024*1024), est_bytes / (1024*1024*1024), item))
        #stream.pbar.refresh()
    async def callback(err, response, url):
      nonlocal received_bytes, received_count, failed_count
      if err is not None:
        #stream.pbar.write('Failed: {!r}: {!r}'.format(str(response.url), response))
        failed_count += 1
        update()
        return
      received_count += 1
      received_bytes += len(response.content)
      update()
      #with BytesIO(response.content) as bio, PIL.Image.open(bio) as image:
      if False:
        #url = str(response.url)
        #stream.pbar.write('Received {size} bytes: {url!r} {image!r}'.format(size=len(response.content), url=url, image=image))
        u = urlparse(url)
        #path = u.netloc + u.path
        path = u.scheme + '/' + u.netloc + u.path
        if u.query:
          path += '?' + u.query
        # #stream.pbar.write(os.path.join(u.netloc, u.path))
        # dim = ('%dx%d' % (image.width, image.height)).ljust(10)
        # fmt = image.format.ljust(5)
        # #stream.pbar.write(dim + fmt + path)
        stream.pbar.write(path)
    #limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
    bar_format = r'{l_bar}{bar}{r_bar}'.format(
        #l_bar='{desc}: {percentage:3.0f}%|',
        #r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]',
        l_bar='{desc}',
        r_bar='| {percentage:3.3f}% [{elapsed}<{remaining}]',
        bar='{bar}'
        )
    async with httpx.AsyncClient(limits=limits) as client:
      for url in shuffled(stream(urls, bar_format=bar_format), start=args.start):
        if received_count + failed_count >= args.maxcount:
          posix._exit(1)
        current_item = url
        if len(dltasks) >= args.concurrency:
          # Wait for some download to finish before adding a new one
          _done, dltasks = await asyncio.wait(dltasks, return_when=asyncio.FIRST_COMPLETED)
        task = process(client, callback, url, stream=stream)
        dltasks.add(loop.create_task(task))
      # Wait for the remaining downloads to finish
      await asyncio.wait(dltasks)


if __name__ == '__main__':
  argv = args.args = sys.argv[1:]
  urls = argv[0] if len(argv) >= 1 and argv[0] else 'https://battle.shawwn.com/danbooru2019-s.txt'
  args.concurrency = int(argv[1]) if len(argv) >= 2 else args.concurrency
  args.shufflesize = int(argv[2]) if len(argv) >= 3 else args.shufflesize
  args.start = int(argv[3]) if len(argv) >= 4 else args.start
  args.maxcount = int(argv[4]) if len(argv) >= 5 else args.maxcount
  activate_mocks()
  loop = asyncio.get_event_loop()
  loop.run_until_complete(main(loop, urls))
