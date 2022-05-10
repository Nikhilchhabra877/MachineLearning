'''
This file shows ,how to read json data and data from the mysql and sqlite database.
'''
import pandas as pd
# fetch data from json file(local)

df  = pd.read_json('/Users/nikhil/Downloads/100-days-of-machine-learning-main/day16 - working-with-json-and-sql/train.json')
print(df.head())
print()

# fetch data from url

df1 = pd.read_json('https://api.exchangerate-api.com/v4/latest/INR')
print(df1.head())

# Read data from sql
import mysql.connector as db

db = db.connect(host='localhost',user='root',password='rootroot',db='datascience')
cur=db.cursor()
cur.execute('use datascience')
#cur.execute('show tables')
cur.execute('select * from location limit 5')
#cur.execute('select * from test')
print(cur.fetchall())

## Read data from sqllite

import sqlite3

conn = sqlite3.connect('/Users/nikhil/Desktop/test.db')
print( "Opened database successfully")
cur = conn.execute("select * from orders")
for i in cur:
    print('order_id'   ,i[0])
    print('order_date', i[1])
    print('com_id', i[2])
    print('sales_id', i[3])
    print('amount', i[4],sep='\t')
conn.close()
print('connection closed successfully')