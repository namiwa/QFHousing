'''
This script scrapes data from unique addressses
'''
import requests
import pandas as pd
import time

# function for singular query
def oneMapQuery(searchVal):
  try: 
    oneMapApi = 'https://developers.onemap.sg/commonapi/search?'
    returnGeom = 'Y'
    getAddrDetails = 'Y'
    pageNum = '1'
    res = requests.get(oneMapApi + 'searchVal=' + searchVal + '&returnGeom=' + returnGeom + '&getAddrDetails=' + getAddrDetails + "&pageNum=" + pageNum, verify=False);
    resJson = res.json();
    print(resJson)
    postal = resJson['results'][0]['POSTAL']
    lat = resJson['results'][0]['LATITUDE']
    long = resJson['results'][0]['LONGTITUDE']
    return [postal, lat, long]
  except Exception as e:
    print("There was an error:", e)
    return ['' , '', '']

# function for running query for main addresses
def main():
  df = pd.read_csv('data/unique_addr.csv')
  store = []
  count = 0
  for index, row in df.iterrows():
    print(row)
    count = count + 1
    searchVal = row['block'] + " " +  row['street_name']
    print(searchVal)
    res = oneMapQuery(searchVal)
    store.append(res)
    if (count > 250):
      count = 0
      time.sleep(60)
  test = pd.DataFrame(store, columns=['postal', 'lat', 'long'])
  test.to_csv('data/test.csv')


def mall_search():
  df = pd.read_csv('data/shopping_malls_may_2020_sg.csv')
  store = []
  count = 0
  for index, row in df.iterrows():
    print(row)
    count = count + 1
    searchVal = row['Malls']
    print(searchVal)
    res = oneMapQuery(searchVal)
    store.append(res)
    time.sleep(1)
  test = pd.DataFrame(store, columns=['postal', 'lat', 'long'])
  df = df.join(test)
  df.to_csv('data/test_malls.csv')

if __name__ == '__main__':
  mall_search()
