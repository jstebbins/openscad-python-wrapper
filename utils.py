import __main__

def get_fnas():
    try:
        fn = __main__.fn
    except AttributeError:
        fn = 1
    try:
        fs = __main__.fs
    except AttributeError:
        fs = 1
    try:
        fa = __main__.fa
    except AttributeError:
        fa = 1

    return (fn, fa, fs)

