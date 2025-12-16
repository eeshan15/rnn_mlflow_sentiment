import pandas as pd
import os



data = {
    "text": [
        "amazing movie great story", "excellent performance loved it", "best film ever", "very good watch",
        "terrible movie waste of time", "bad acting poor script", "worst film ever", "boring and slow",
        "loved the acting", "plot was terrible", "great cast but bad story", "waste of money","Not worth it slept whole the time"
    ],
    "label": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0 ,0]
}

df = pd.DataFrame(data)
df.to_csv("data/raw_data.csv",index = False)
print("Raw data has been created !! ")