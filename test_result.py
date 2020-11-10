import pandas as pd

df = pd.read_excel("data/result.xlsx")
address1s = df["目标地址"]
address2s = df["被比较地址"]
similaritys = df["相似度"]
results=[]
for address1,address2,simi in zip(address1s,address2s,similaritys):
    if address1 in address2 or address2 in address1 and address2!=address1:
        results.append([address1,address2,simi])

print(results)
df = pd.DataFrame(results)
df.to_excel("data/result_test.xlsx")

