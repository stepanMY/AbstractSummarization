import asyncio
import aiohttp
import nest_asyncio
nest_asyncio.apply()


class AsyncParser:
    """
    Class that parses webpages asynchronously
    """
    def __init__(self,
                 urls,
                 process,
                 n_connections,
                 n_retries=5,
                 retrywait=0.5):
        """
        @param urls: list, list of web addresses to parse
        @param process: function, function that accepts url and page text, handles it
               and returns result
        @param n_connections: int, limit of simultaneously opened connections
        @param n_retries: int, number of possible retries
        @param retrywait: int, seconds to wait before retrying to reach url
        """
        self.urls = urls
        self.process = process
        self.n_connections = n_connections
        self.n_retries = n_retries
        self.retrywait = retrywait
        self.ranflag = False
        self.notreachedurls = []

    async def geturl(self, session, url):
        """
        Process one url

        @param session: aiohttp.ClientSession, session client to use
        @param url: string, url to process
        """
        async with session.get(url) as resp:
            if resp.status == 200:
                text_ = await resp.text()
                return self.process(url, text_)
        async with session.get(url) as resp:
            for _ in range(self.n_retries):
                if resp.status == 200:
                    text_ = await resp.text()
                    return self.process(url, text_)
            self.notreachedurls.append(url)
            return None

    async def geturls(self):
        """
        Process all urls
        """
        connector = aiohttp.TCPConnector(limit=self.n_connections)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for url in self.urls:
                tasks.append(asyncio.ensure_future(self.geturl(session, url)))

            results = await asyncio.gather(*tasks)
            return results

    def parse(self):
        """
        Start parsing job
        """
        self.ranflag = True
        return asyncio.run(self.geturls())

    def notreached(self):
        """
        Return list of not reached urls
        """
        if self.ranflag:
            raise AttributeError('Didn\'t parse')
        return self.notreachedurls
