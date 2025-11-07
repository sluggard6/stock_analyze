import akshare as ak
import numpy as np
from datetime import datetime, timedelta
from ollama import Client
import os



# 配置参数
STOCK_CODE = "sz000002"
# 股票代码，可修改为目标股票
client = Client(host="https://ollama.com",
                headers={'Authorization': 'b585f727c42d4b058eb559fc1f286b00.WWV4QeWiZprlw7yR0tgpqABM'}
                )

# 获取近三年的日期范围
# end_date = datetime.now().strftime("%Y%m%d")
# start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")

# 1. 使用akshare获取股票数据
def get_stock_data(stock_code, start_date, end_date):
    stock_df = ak.stock_zh_a_daily(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
    df = stock_df[['date', 'close', 'volume']].copy()

    # 原有指标
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    df['MA120'] = df['close'].rolling(120).mean()

    # 新增指标
    df['VMA5'] = df['volume'].rolling(5).mean()  # 成交量5日均线
    df['VMA20'] = df['volume'].rolling(20).mean() # 成交量20日均线
    df['VMA60'] = df['volume'].rolling(60).mean() # 成交量60日均线
    df['VMA120'] = df['volume'].rolling(120).mean() # 成交量120日均线

    # 布林带计算
    df['BOLL_MID'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['BOLL_UPPER'] = df['BOLL_MID'] + 2*std
    df['BOLL_LOWER'] = df['BOLL_MID'] - 2*std

    # RSI计算
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # 量价背离检测
    df['PRICE_HIGHER'] = df['close'] > df['close'].shift(1)
    df['VOLUME_LOWER'] = df['volume'] < df['volume'].shift(1)
    df['Volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std() * np.sqrt(20)
    df['Volatility'] = df['Volatility'].fillna(0)  # 处理初始NaN值

    # MACD计算
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']

    # KDJ计算
    df['K'] = df['close'].ewm(span=9, adjust=False).mean()
    df['D'] = df['K'].ewm(span=3, adjust=False).mean()
    df['J'] = 3*df['K'] - 2*df['D']
    df['KDJ'] = df[['K', 'D', 'J']].apply(lambda x: ','.join(map(str, x)), axis=1)
    return df

# 2. 准备发送给大模型的分析提示
def prepare_analysis_prompt(df):
    """准备分析提示词"""
    # 获取最新数据
    latest_data = df.iloc[-1].to_dict()
    # 数据摘要
    data_summary = f"股票代码: {STOCK_CODE}\n"
    data_summary += f"数据范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}\n"
    data_summary += f"最新收盘价: {latest_data['close']:.2f}元\n"
    data_summary += f"近期波动率: {latest_data['Volatility']:.2%}\n"
    data_summary += f"5日均线: {latest_data['MA5']:.2f}元\n"
    data_summary += f"20日均线: {latest_data['MA20']:.2f}元\n"
    data_summary += f"60日均线: {latest_data['MA60']:.2f}元\n"
    data_summary += f"120日均线: {latest_data['MA120']:.2f}元\n"
    data_summary += f"5日量均价: {latest_data['VMA5']:.2f}元\n"
    data_summary += f"20日量均价: {latest_data['VMA20']:.2f}元\n"
    data_summary += f"60日量均价: {latest_data['VMA60']:.2f}元\n"
    data_summary += f"120日量均价: {latest_data['VMA120']:.2f}元\n"
    data_summary += f"布林带中轨: {latest_data['BOLL_MID']:.2f}元\n"
    data_summary += f"布林带上限: {latest_data['BOLL_UPPER']:.2f}元\n"
    data_summary += f"布林带下限: {latest_data['BOLL_LOWER']:.2f}元\n"
    data_summary += f"RSI: {latest_data['RSI']:.2f}\n"
    data_summary += f"价格高于前一日: {latest_data['PRICE_HIGHER']}\n"
    data_summary += f"成交量低于前一日: {latest_data['VOLUME_LOWER']}\n"
    data_summary += f"价格高于前一日: {latest_data['PRICE_HIGHER']}\n"
    data_summary += f"成交量低于前一日: {latest_data['VOLUME_LOWER']}\n"
    data_summary += f"量价背离次数: {sum(df['PRICE_HIGHER'] & df['VOLUME_LOWER'])}次\n"
    data_summary += f"MACD: {latest_data['MACD']:.2f}\n"
    data_summary += f"MACD信号线: {latest_data['MACD_SIGNAL']:.2f}\n"
    data_summary += f"MACD直方图: {latest_data['MACD_HIST']:.2f}\n"
    data_summary += f"KDJ: {latest_data['KDJ']}\n"


    # 分析要求
    prompt = f"以下是{STOCK_CODE}股票近三年的收盘价数据及技术指标:\n"
    prompt += f"{data_summary}\n\n"
    prompt += "请使用多种策略对该股票进行分析，包括但不限于：\n"
    prompt += "1. 趋势分析：判断当前处于上升、下降或横盘趋势\n"
    prompt += "2. 均线分析：分析短期、中期和长期均线的排列关系\n"
    prompt += "3. 支撑压力分析：基于历史价格判断关键支撑位和压力位\n"
    prompt += "4. 波动性分析：评估当前波动率处于历史什么水平\n"
    prompt += "5. 未来走势预测：基于历史数据给出短期和中期走势预测\n"
    prompt += "6. 投资建议：给出买入、持有或卖出的建议及理由\n"
    prompt += "7. 量价背离检测：统计价格上涨但成交量下降的天数\n"
    prompt += "8. 相对强弱指标：对比行业指数和大盘走势评估股票强弱\n"
    prompt += "9. 布林带分析：识别价格波动区间和突破信号\n"
    prompt += "10. 资金流向分析：监控主力资金和大单交易动向\n"
    prompt += "11. 市场情绪分析：结合新闻舆情和行业政策评估影响\n"
    prompt += "12. 风险回报比：计算潜在收益与止损空间的比例\n"
    prompt += "13. 技术指标分析：结合MACD、RSI、KDJ等指标综合判断\n"
    prompt += "14. 财务指标分析：评估市盈率、市净率等财务指标\n"
    prompt += "15. 行业分析：研究同行业其他公司表现和竞争态势\n"
    prompt += "16. 资金面分析：关注央行政策、货币政策对股市的影响\n"
    prompt += "17. 市场容量分析：评估行业增长潜力和市场规模\n"
    prompt += "18. 盈利能力分析：评估公司的盈利能力和盈利前景\n"
    prompt += "19. 偿债能力分析：评估公司的偿债能力和风险水平\n"
    prompt += "20. 股息政策分析：评估公司的股息支付能力和政策\n"
    prompt += "21. 股东结构分析：评估公司股权分散程度和机构持仓\n"
    prompt += "22. 股东变动分析：关注公司股权变动和资本运作\n"
    prompt += "23. 财务报表分析：评估公司的财务状况和盈利能力\n"
    prompt += "请按以下格式组织分析结果：\n"
    prompt += "【策略名称】分析内容...\n"
    prompt += "[策略间验证] 说明不同策略结论的相互印证关系\n"
    prompt += "[综合结论] 基于多策略分析给出最终操作建议"
    prompt += "请使用中文回答。"

    return prompt

# 3. 调用Ollama Turbo大模型
def call_siliconflow_model(prompt):
    messages = [{'role': 'user', 'content': prompt}]
    resp = client.chat(model='gpt-oss:120b', messages=messages, stream=False)
    return resp['message']['content']


def all_in_one(stock_code):
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y%m%d")
    df = get_stock_data(STOCK_CODE, start_date, end_date)
    prompt = prepare_analysis_prompt(df)
    analysis_result = call_siliconflow_model(prompt)
    return analysis_result

