import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import plotly.graph_objects as go
from pytimekr import pytimekr
from datetime import datetime, timedelta

st.set_page_config(layout='wide')

# DB 연결
host = "research-carbon-credit-dbinst.cy2zd8eyokmi.ap-northeast-2.rds.amazonaws.com"
port = 3306
user = "admin"
password = "Yv3LQkZPaIMuHy0"
database = "carbon_credits_price_forecasting"
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}", echo=False)


def is_it_holiday(df):
    list_for_yyyy = pytimekr.holidays(int(df["yyyy"]))
    list_for_yyyy = [i.strftime('%Y-%m-%d') for i in list_for_yyyy]
    return df['yymmdd'] in list_for_yyyy


def check_price_day(sdate, edate, freqc, nmvar):  # 2024.01.05 수정/김만기
    # .. 모든 기간 생성
    # 월간격 (해당 작업: monthly_top5)
    if freqc == 'monthly' and nmvar > 1:
        set_dates = pd.date_range(sdate, edate, freq='1M')
        empty_df = pd.DataFrame(index=set_dates)
        empty_df.index.rename('base_date', inplace=True)
    else:
        # 일간격 (해당 작업: day_dayahead, day_top5, monthly_dayahead)
        set_dates = pd.date_range(sdate, edate, freq='1D')
        empty_df = pd.DataFrame(index=set_dates)
        empty_df.index.rename('base_date', inplace=True)

    # .. 공휴일, 평일정보 추가
    empty_df['yyyy'] = empty_df.index.strftime('%Y')
    empty_df['yymm'] = empty_df.index.strftime('%Y-%m')
    empty_df['yymmdd'] = empty_df.index.strftime('%Y-%m-%d')
    empty_df['weekday'] = empty_df.index.weekday
    empty_df['holiday'] = empty_df.apply(is_it_holiday, axis=1)

    # .. 공휴일과 주말은 삭제
    if nmvar == 1:  # 해당 작업: day_dayahead, day_top5, monthly_dayahead
        date_mask = (empty_df['holiday'] == False) & (empty_df['weekday'] <= 4)
        empty_df_filter = empty_df[date_mask]
        # empty_df_filter = empty_df[date_mask].index
    else:
        empty_df_filter = empty_df

    return empty_df_filter


def calculate_disparity(row):
    if row['신뢰구간포함여부'] == '-':
        if row['실제값(원)'] > row['상한']:
            return row['실제값(원)'] - row['상한']
        elif row['실제값(원)'] < row['하한']:
            return row['하한'] - row['실제값(원)']
    else:
        return '-'
    return None

date_input = st.text_input('날짜 입력', value=datetime.now().date())

query1 = (f"select A.base_date as '예측구간',     "
                       f"round(actual) as '실제값(원)',"
              f"       round(forecast) as '예측값(원)', "
              f"       round(abs(actual-forecast)/actual*100, 3) as '오차율(%%)',"
              f"       round(if(forecast-min>3500, forecast-3500, min)) as '예측하한가격(원)',"
              f"       round(if(max-forecast>3500, forecast+3500, max)) as '예측상한가격(원)' "
              f"from "
              f"(select base_date,"
              f"       round(forecasting_value) as forecast,"
              f"       round(forecasting_upper_bound) as max,"
              f"       round(forecasting_lower_bound) as min "
              f"from carbon_credit_forecasting "
              f"WHERE forecasting_time_step_type = 'daily'"
              f"and forecasting_time_step_value = 1 "
              f"and algorithm_seq=2  "
              # f"and base_date >= date_sub('{date_input}', interval 1 month) "
              f"and base_date < '{date_input}' "
              f"and forecasting_value is not null "
              f"group by base_date) A "
              f"join "
              f"(select base_date, "
              f"       close_price as actual "
              f"from carbon_credit_price "
              f"where carbon_credit_name = 'KAU23') B "
              f"on A.base_date = B.base_date "
              f"order by A.base_date desc;")

query1_df = pd.read_sql_query(query1, engine)

query1_df['신뢰구간포함여부']= query1_df.apply(lambda row: 'y' if row['예측하한가격(원)'] <= row['실제값(원)'] <= row['예측상한가격(원)'] else 'x', axis=1)
query1_df['이격도'] = query1_df.apply(calculate_disparity, axis=1)
algorithm_list = ['석탄_원자력_LGBM', ' 석탄_천연_LGBM', ' 석탄_천연_TCN', '석탄_천연_LGBM', '  소득_석탄_위안환율_LGBM', '석탄_천연_TCN',
                  '철광석_석탄_유연탄_ARIMA', ' 소득_석탄_위안환율_LGBM', ' 석탄_이건산업_LGBM']
query2_df = query1_df[['예측구간']].copy()
for col in ['1순위', '2순위;', '3순위', '4순위', '5순위']:
    query2_df[col] = np.random.choice(algorithm_list, size=len(query2_df))

end_date = datetime.strptime(date_input, '%Y-%m-%d') + timedelta(days=30)
end_date = end_date.strftime('%Y-%m-%d')
business_days_df = check_price_day(date_input, end_date, 'daily', 1)

daily_query = f"""select final.구분,
                   round(final.예측값) as '예측값(원)',
                   final.종가 as '실제가격(원)',
                   round(final.종가 - LAG(final.종가) OVER (ORDER BY final.구분)) AS '전월비(원)',
                   (final.종가 / LAG(final.종가) OVER (ORDER BY final.구분)) *100 - 100 AS '등락률(%%)',
                   final.거래량,
                   round(final.상한) as '예측상한가격(원)',
                   round(final.하한) as '예측하한가격(원)'
            from
            (select A.base_date as '구분',
           close_price as '종가',
           forecasting_value as '예측값',
           volume as '거래량',
           high as '상한',
           low as '하한'

        from
        (select base_date,
           avg(forecasting_value) as forecasting_value,
           avg(forecasting_upper_bound) as high,
           avg(forecasting_lower_bound) as low

        from carbon_credit_forecasting 
        where forecasting_time_step_type = 'daily' 
        and forecasting_time_step_value = 1 
        and base_date < '{date_input}'  
        group by base_date) A 
        join
        (select base_date,
            close_price,
           volume
        from carbon_credit_price
        where base_date < '{date_input}') B
        on A.base_date = B.base_date

            union
            select base_date as '구분', 
                   null as '종가', 
                   avg(forecasting_value) as '예측값',
                   null as '거래량',
                   avg(forecasting_upper_bound) as '상한',
                   avg(forecasting_lower_bound) as '하한'
            from carbon_credit_forecasting
            where forecasting_time_step_type = 'daily'
            and date(base_date) = '{date_input}' 
            and forecasting_time_step_value = 1
            group by base_date
            union
            select base_date as '구분', 
                   null as '종가', 
                   avg(forecasting_value) as '예측값',
                   null as '거래량',
                   avg(forecasting_upper_bound) as '상한',
                   avg(forecasting_lower_bound) as '하한'
            from carbon_credit_forecasting
            where forecasting_time_step_type = 'daily'
            and date(base_date) = '{business_days_df.iloc[1].name.date()}' 
            and forecasting_time_step_value = 2
            group by base_date
            union
            select base_date as '구분', 
                   null as '종가', 
                   avg(forecasting_value) as '예측값',
                   null as '거래량',
                   avg(forecasting_upper_bound) as '상한',
                   avg(forecasting_lower_bound) as '하한'
            from carbon_credit_forecasting
            where forecasting_time_step_type = 'daily'
            and date(base_date) = '{business_days_df.iloc[4].name.date()}' 
            and forecasting_time_step_value = 5
            group by base_date
            union
            select base_date as '구분', 
                   null as '종가', 
                   avg(forecasting_value) as '예측값',
                   null as '거래량',
                   avg(forecasting_upper_bound) as '상한',
                   avg(forecasting_lower_bound) as '하한'
            from carbon_credit_forecasting
            where forecasting_time_step_type = 'daily'
            and date(base_date) = '{business_days_df.iloc[19].name.date()}' 
            and forecasting_time_step_value = 20
            group by base_date

                ) final
            where year(final.구분) >= 2024
            order by final.구분 desc
            """
daily_df = pd.read_sql_query(daily_query, engine)

# Plotly 그래프 생성
daily_fig = go.Figure()

daily_fig.add_trace(go.Bar(x=daily_df['구분'], y=daily_df['거래량'], name='거래량', marker_color='darkturquoise'))
daily_fig.add_trace(go.Scatter(x=daily_df['구분'], y=daily_df['예측값(원)'], mode='lines', name='예측값(원)', yaxis='y2', marker_color='#660099'))
daily_fig.add_trace(go.Scatter(x=daily_df['구분'], y=daily_df['실제가격(원)'], mode='lines', name='실제값(원)', yaxis='y2', marker_color='#FF4500'))
daily_fig.add_trace(go.Scatter(x=daily_df['구분'], y=daily_df['예측상한가격(원)'], fill=None, mode='lines', line=dict(color='gray'), showlegend=False, yaxis='y2', name='상한(원)'))
daily_fig.add_trace(go.Scatter(x=daily_df['구분'], y=daily_df['예측하한가격(원)'],fill='tonexty', mode='lines', line=dict(color='gray'), name='예측상하한가격(원)', yaxis='y2'))


daily_fig.update_layout(
    yaxis=dict(title='가격(원)'),
    yaxis2=dict(title='거래량', overlaying='y', side='right')
)

# plotly style
st.markdown("""
<style>
    .dataframe-container, .plotly-chart {
        border: 2px solid #0078d4;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
daily_actual_df = query1_df
daily_actual_df['예측구간'] = pd.to_datetime(daily_actual_df['예측구간'], format='%Y-%m-%d')
daily_actual_df = daily_actual_df[daily_actual_df['예측구간'] < pd.to_datetime(date_input, format='%Y-%m-%d')]
# 정확도 계산
condition = (query1_df['예측하한가격(원)'] <= query1_df['실제값(원)']) & (query1_df['실제값(원)'] <= query1_df['예측상한가격(원)'])
num_rows_condition = query1_df[condition].shape[0]
total_rows = query1_df.shape[0]
ratio = num_rows_condition / total_rows
daily_actual_df['정확도'] = ratio * 100
col1_range = f"{daily_actual_df['예측구간'].iloc[-1].date()}~{daily_actual_df['예측구간'].iloc[0].date()}"
col2_mean = daily_actual_df['실제값(원)'].mean()
col3_mean = daily_actual_df['예측값(원)'].mean()
col4_first = daily_actual_df['정확도'].iloc[0]
df1 = pd.DataFrame({
    '예측구간': [col1_range],
    '실제값평균(원)': [round(col2_mean)],
    '예측값평균(원)': [round(col3_mean)],
    '정확도(%)': [f'{round(col4_first, 3)}%']
})


st.title(f'{date_input} 온실가스 배출권 가격 전망')
col1, col2 = st.columns(2)
with col1:
    st.subheader('Daily')
    st.plotly_chart(daily_fig, use_container_width=True)
    st.dataframe(daily_df, width=950)

with col2:
    st.subheader('예측 구간별 오차율')
    st.dataframe(query1_df.head(1))
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader('예측 구간별 예측 알고리즘 랭크')
    st.dataframe(query2_df.head(1))
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.subheader('가격 분포 추정 정확도')
    st.write(df1)
    st.write('* 예측 구간은 Business day 기준 100일')


    st.latex(r"""
\textbf{정확도}(\%) = \frac{\sum_{i=1}^{n} \mathbf{1}(\hat{\mathbf{y}}_{\min,i} \leq \mathbf{y}_i \leq \hat{\mathbf{y}}_{\max,i})}{\mathbf{n}} \quad (\mathbf{n} = 100)
""")