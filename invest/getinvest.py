
#-*- coding: utf-8 -*-
'''
Created on 2016年8月9日

@author: bmexue  bmexue@163.com
'''


from urllib import request
from bs4 import BeautifulSoup
from mylog import MyLog as mylog
import time



class Stock(object):
    count = 0  # 股票数量
    urlnode = None  #股票url
    title = None    #股票名称
    price = 0     #股票价格
    money = 0
    rmb_hk_dl = 0

g_stock = Stock()

class GetTiebaInfo(object):
    def __init__(self,url):
        self.url = url
        self.log = mylog()
        self.pageSum = 5
        self.spider(self.url)
       

    def spider(self, url):
        items = []
        htmlContent = self.getResponseContent(url)
        soup = BeautifulSoup(htmlContent, 'lxml')
        stock = Stock()

        stock.title = soup.find('div', attrs={'class':'stock-name'}).get_text().strip()
        stock.price = soup.find('div', attrs={'class':'stock-current'}).get_text().strip()
        plen = len(stock.price)
        stock.price=  stock.price[1:plen]
        print ( stock.title)
        print (stock.price)
        g_stock = stock

    
    def pipelinesstock (self,stocks):
        a= 1

    def pipelines(self, items):
        fileName = u'百度贴吧_权利的游戏.txt'.encode('GBK')
        with open(fileName, 'w') as fp:
            for item in items:
                fp.write('title:%s \t author:%s \t firstTime:%s \n content:%s \n return:%s \n lastAuthor:%s \t lastTime:%s \n\n\n\n' 
                         %(item.title.encode('utf8'),item.firstAuthor.encode('utf8'),item.firstTime.encode('utf8'),item.content.encode('utf8'),item.reNum.encode('utf8'),item.lastAuthor.encode('utf8'),item.lastTime.encode('utf8')))
                self.log.info(u'标题为<<%s>>的项输入到"%s"成功' %(item.title, fileName.decode('GBK')))

    def getResponseContent(self, url):
        '''这里单独使用一个函数返回页面返回值，是为了后期方便的加入proxy和headers等
        '''
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)')]
        response = opener.open(url)
        page = response.read()
        return page

def GetStock(url):
    opener = request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)')]
    response = opener.open(url)
    htmlContent = response.read()

    soup = BeautifulSoup(htmlContent, 'lxml')
    title = soup.find('div', attrs={'class':'stock-name'}).get_text().strip()
    price = soup.find('div', attrs={'class':'stock-current'}).get_text().strip()
    plen = len(price)
    index = 0
    while index < plen:
       strheder = price[index]
       if strheder.isalnum() == True:
            price=  price[index:plen]
            break
       index = index + 1
    return title, price

def position_cmp(tmpx, tmpy):
    #tmpx = float(nodex.count) * float(nodex.price)
    #tmpy = float(nodey.count) * float(nodey.price)
    if tmpx > tmpy:
        return -1
    if tmpx < tmpy:
        return 1
    return 0

if __name__ == '__main__':
    money = 120000
    stocks = []



    stock9 =Stock()
    stock9.urlnode = 'SH600104'
    stock9.count = 3000
    stocks.append(stock9)

    stock1 =Stock()
    stock1.urlnode = 'SZ000651'
    stock1.count = 3500
    stocks.append(stock1)

    stockt =Stock()
    stockt.urlnode = 'SH601939'
    stockt.count = 12900
    stocks.append(stockt)

    stock2 =Stock()
    stock2.urlnode = 'SZ000333'
    stock2.count = 2000
    stocks.append(stock2)

    stock3 =Stock()
    stock3.urlnode = 'SH513050'
    stock3.count = 72000
    stocks.append(stock3)

    stock4 =Stock()
    stock4.urlnode = 'SH600585'
    stock4.count = 3000
    stocks.append(stock4)

    stock5 =Stock()
    stock5.urlnode = 'SH600009'
    stock5.count = 900
    stocks.append(stock5)

    stockt858 =Stock()
    stockt858.urlnode = 'SZ000858'
    stockt858.count = 600
    stocks.append(stockt858)

    stockt887 =Stock()
    stockt887.urlnode = 'SH600887'
    stockt887.count = 1100
    stocks.append(stockt887)

    stockt0002 =Stock()
    stockt0002.urlnode = 'SZ000002'
    stockt0002.count = 600
    stocks.append(stockt0002)



    index = 0
    totle = money
    for stock in stocks:
        urlroot = 'https://xueqiu.com/S/'
        url = urlroot+stock.urlnode

        title,price = GetStock(url)
        stocks[index].title   = title
        stocks[index].price   = price
        print ('stock %f  %s' %(stocks[index].count,stocks[index].price))
        tmp = float(stocks[index].count) * float(stocks[index].price)
        if stock.rmb_hk_dl == 1 :
            tmp = tmp * 0.81
        stocks[index].money = tmp
        totle += tmp
        index = index + 1
        time.sleep(0.1)
    
    stocks = sorted(stocks, key=lambda x:x.money,reverse=True)

    print('Totle: %f' % (totle))

    print ('Stock name             count       price       position')
    print (' %s %.0f                 %.2f%%'% ("Cash",money,100*money/totle))
    for node in stocks:
        tmp = float(node.count) * float(node.price) * 100
        print (' %20s %.0f  %s       %.2f%%'% (node.title,node.count,node.price,tmp/totle))
