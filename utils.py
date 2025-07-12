import __main__
from dataclasses import dataclass
import time

def get_fnas():
    try:
        fn = __main__.fn
    except AttributeError:
        fn = None
    try:
        fs = __main__.fs
    except AttributeError:
        fs = 5
    try:
        fa = __main__.fa
    except AttributeError:
        fa = 5

    return fn, fa, fs

prof_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
prof_last  = prof_start
prof_dict  = dict()
profile = True

@dataclass()
class ProfAccumulate():
    lap_start : int = None
    ellapse   : int = None

def prof_lap_start(key):
    global prof_dict

    if key not in prof_dict:
        prof_dict[key] = ProfAccumulate(lap_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC), ellapse = 0)
    prof_dict[key].lap_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

def prof_lap_pause(key):
    global prof_dict

    now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    prof_dict[key].ellapse += now - prof_dict[key].lap_start

def prof_lap_finish(key, msg=""):
    global prof_dict

    if not profile: return
    print(f"{msg} - {key} ellapsed: {prof_dict[key].ellapse / (1000*1000)}ms")
    del prof_dict[key]

def prof_time(msg="", final=False):
    global prof_last
    global prof_start
    global verbose

    if not profile: return
    last = prof_start if final else prof_last
    now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    print(f"{msg} - ellapsed: {(now - last) / (1000*1000)}ms")
    prof_last = now
