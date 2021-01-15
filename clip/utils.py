import requests
import tqdm
import os
from urllib.parse import urlparse
from contextlib import ExitStack
import multidict

ITER_CHUNK_SIZE = 512

def is_remote_url(url):
  info = urlparse(url)
  return bool(info.scheme and info.netloc)

def fetch_url_content_length(url):
  if is_remote_url(url):
    head = requests.head(url, headers={'Accept-Encoding': None})
    return int(head.headers['Content-Length'])
  else:
    return os.path.getsize(url)

def fetch_url_streaming(url):
  if is_remote_url(url):
    return requests.get(url, stream=True)
  else:
    return open(url, 'rb')

def iter_lines(
    iterator_or_url,
    total_bytes=None,
    decode_unicode=True,
    chunk_size=ITER_CHUNK_SIZE,
    progress_bar=True,
    dynamic_ncols=True,
    **tqdm_options):
    """Iterates over the response data, one line at a time.  When
    stream=True is set on the request, this avoids reading the
    content at once into memory for large responses.

    .. note:: This method is not reentrant safe.
    """

    url = None

    with ExitStack() as stack:

        if isinstance(iterator_or_url, str):
            url = iterator_or_url
            if total_bytes is None:
                total_bytes = fetch_url_content_length(url)
            iterator = fetch_url_streaming(url)

        stack.enter_context(iterator)

        if hasattr(iterator, 'iter_content'):
            iterator = iterator.iter_content(chunk_size=chunk_size, decode_unicode=False)
            #stack.enter_context(iterator)

        pending = None
        disable_progress_bar = total_bytes is None or not progress_bar

        with tqdm.trange(total_bytes, disable=disable_progress_bar, dynamic_ncols=dynamic_ncols, **tqdm_options) as pbar:

          def update(line):
              pbar.update(len(line))
              line = line.splitlines()
              assert len(line) == 1
              line = line[0]
              return line

          for chunk in iterator:
              pbar.update(len(chunk))

              if isinstance(chunk, bytes):
                  chunk = chunk.decode('utf-8')

              if pending is not None:
                  chunk = pending + chunk

              lines = chunk.splitlines(keepends=True)

              if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                  pending = lines.pop()
              else:
                  pending = None

              for line in lines:
                  yield update(line)

          if pending is not None:
              yield update(pending)


class LineStream(ExitStack):
    def close(self):
        super().close()
        getattr(self, 'lines', []).clear()
        self.closed = True

    def __call__(
        self,
        iterator_or_url,
        total_bytes=None,
        decode_unicode=True,
        chunk_size=ITER_CHUNK_SIZE,
        progress_bar=True,
        dynamic_ncols=False,
        unit_scale=1,
        **tqdm_options):
        """Iterates over the response data, one line at a time.  When
        stream=True is set on the request, this avoids reading the
        content at once into memory for large responses.

        .. note:: This method is not reentrant safe.
        """

        if getattr(self, 'started', False):
            raise RuntimeError("LineStream can't be started more than once")
        self.started = True
        self.url = None
        self.closed = False
        self.lines = []
        self.decode_unicode = decode_unicode
        self.sizes = multidict.MultiDict()

        if isinstance(iterator_or_url, str):
            self.url = iterator_or_url
            if total_bytes is None:
                total_bytes = fetch_url_content_length(self.url)
            self.iterator = fetch_url_streaming(self.url)

        self.enter_context(self.iterator)

        if hasattr(self.iterator, 'iter_content'):
            self.iterator = self.iterator.iter_content(chunk_size=chunk_size, decode_unicode=False)

        self.pending = None
        disable_progress_bar = total_bytes is None or not progress_bar

        self.pbar = tqdm.tqdm(total=total_bytes, disable=disable_progress_bar, dynamic_ncols=dynamic_ncols, unit_scale=unit_scale, **tqdm_options)
        self.enter_context(self.pbar)

        for chunk in self.iterator:

            if self.pending is not None:
                chunk = self.pending + chunk

            lines = chunk.splitlines(keepends=True)

            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                self.pending = lines.pop()
            else:
                self.pending = None

            self.lines = lines
            while True:
                try:
                    line = self.lines.pop(0)
                except IndexError:
                    if self.closed:
                        return
                    else:
                        break
                yield self.update(line)

        if self.pending is not None:
            yield self.update(self.pending)


    def update(self, line):
        n = len(line)
        line = line.splitlines()
        assert len(line) == 1
        line = line[0]
        if isinstance(line, bytes) and self.decode_unicode:
            line = line.decode('utf-8')
        self.sizes.add(line, n)
        return line

    def finish(self, line):
        n = self.sizes.pop(line)
        self.pbar.update(n)


# Python 3.5 backport. Is there a more elegant way to get a nullcontext?
import abc


class AbstractContextManager(abc.ABC):
  """An abstract base class for context managers."""

  def __enter__(self):
    """Return `self` upon entering the runtime context."""
    return self

  @abc.abstractmethod
  def __exit__(self, exc_type, exc_value, traceback):
    """Raise any exception triggered within the runtime context."""
    return None


class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing.
    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:
    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


class ProgressBar(AbstractContextManager):
  def __init__(self, total_size, disable=False, **kws):
    if not disable:
      if not isinstance(total_size, int):
        raise ValueError("Expected total_size to be set for progress bar")
    self.total_size = total_size
    self.disable = disable
    #self.pbar = tqdm.trange(total_size, **kws) if not disable else nullcontext()
    self.pbar = tqdm.trange(total_size, disable=disable, **kws)

  def __enter__(self):
    self.pbar.__enter__()
    return self

  def __exit__(self, *excinfo):
    return self.pbar.__exit__(*excinfo)

  def update(self, n):
    if not self.disable:
      return self.pbar.update(n)
