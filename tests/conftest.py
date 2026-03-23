import asyncio
import inspect


def pytest_pyfunc_call(pyfuncitem):
    test_fn = getattr(pyfuncitem, "obj", None)
    if test_fn is None:
        return None
    if not inspect.iscoroutinefunction(test_fn):
        return None

    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_fn(**kwargs))
    finally:
        try:
            loop.close()
        finally:
            asyncio.set_event_loop(None)
    return True
