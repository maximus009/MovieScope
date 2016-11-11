import glob
import json
import re
import xlwt
import uuid
import os
import pandas as pd
from pymongo import MongoClient
count=1

#Mongo Collection creation

mng_client=MongoClient('localhost',27017)
db_cm=mng_client.project
collection=db_cm.dataset_test
#coll = mng_client["Project"]["Dataset"]

## Adding rows to Mongo Collection
docs=[]
genre=["action","animation","drama","fantasy","horror","romance"]
data=pd.DataFrame(columns=list(['index','name','genre','format']))
prevdir = os.getcwd()
for g in genre:
    path=".\dataset\\train\\"+g
    print path
    os.chdir(path)
    for filename in ("*.mp4", "*.webm"):
        title=glob.glob(filename)
        format1=filename.split(".")[1]
        for i in range(0,(len(title))):
            title[(i)] = unicode(title[(i)], errors='ignore')
            data.loc[count]=[int(count),title[(i)],g,format1]
            count=int(count+1)
    os.chdir(prevdir)
print data.head()
#data_json=json.loads(data.to_json(orient='records'))

collection.insert_many(data.to_dict(orient='records'))
print(list(collection.find()))
