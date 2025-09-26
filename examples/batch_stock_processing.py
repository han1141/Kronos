import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="mps:0", max_context=512)

# 3. Prepare Data
df = pd.read_csv(".csv/history.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

lookback = 400
pred_len = 120
x_start_idx = -lookback - pred_len
x_end_idx = -pred_len
x_df = df.iloc[x_start_idx:x_end_idx][['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.iloc[x_start_idx:x_end_idx]['timestamps']
y_timestamp = df.iloc[-pred_len:]['timestamps']

# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=3,
    verbose=True
)

# 5. Add timestamps to prediction data
# 确保预测数据包含时间戳
if 'timestamps' not in pred_df.columns:
    pred_df['timestamps'] = y_timestamp.values

# 6. Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback+pred_len-1]

# 7. Export Results to JSON
json_filename = "json/prediction_a_share_results.json"
export_data = {
    "historical_data": kline_df.to_dict('records'),
    "prediction_data": pred_df.to_dict('records')
}

with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

print(f"历史数据和预测结果已导出到: {json_filename}")

# 8. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())
