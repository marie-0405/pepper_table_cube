# coding: UTF-8
import time
from datetime import datetime
from functools import wraps

def time_recorder(func):
  @wraps(func)
  def new_function(*args, **kwargs):
    start_at = time.time()
    start_str = datetime.fromtimestamp(start_at).strftime('%Y-%m-%d %H:%I:%S')

    print('Started func:', func.__name__, '[' + start_str + ']')

    result = func(*args, **kwargs)

    end_at = time.time()
    end_str = datetime.fromtimestamp(end_at).strftime('%Y-%m-%d %H:%I:%S')
    time_taken = end_at - start_at
    print('Finished func:', func.__name__, 'took', '{:.3f}'.format(time_taken), 'sec[' + end_str + ']')

    return result
  return new_function

@time_recorder
def wait_method(sec):
    """サンプルメソッド。引数の間待ち、「Done!」をprintする"""
    time.sleep(sec)

    print('Done!')


if __name__ == '__main__':
    wait_method(3)