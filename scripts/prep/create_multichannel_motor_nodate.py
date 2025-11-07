import pandas as pd
from pathlib import Path

src = Path("/home/ubuntu/hsz/OpenLTM/datasets/永济电机轴承数据集/算法库测试/test/1370_电腐蚀_test.csv")
out = Path("./datasets/1370_电腐蚀.csv")

vals = pd.read_csv(src, header=None, names=["v"]).squeeze().astype(float)
df = pd.DataFrame({"ch1": vals.values, "ch2": vals.values})
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, float_format="%.8f")
print("Saved:", out)