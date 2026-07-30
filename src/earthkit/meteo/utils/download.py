def simple_download(url, target):
    import requests

    r = requests.get(url, allow_redirects=True)
    r.raise_for_status()
    open(target, "wb").write(r.content)

    # import urllib.request
    # urllib.request.urlretrieve(url, target)
