import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
from scipy import stats


def load_data(course: str, student_id: str, email: str) -> tuple:
    if course == '문제 해결 프로그래밍':
        path = './data/problem-solving.csv'
    elif course == '알고리즘':
        path = './data/algorithm.csv'
    elif course == '데이터베이스':
        path = './data/database.csv'
    else:
        raise ValueError('해당하는 과목 성적이 없습니다.')

    try:
        grade = pd.read_csv(path)
        ind = grade.loc[lambda x: (x['student_id'] == int(student_id)) & (x['email'] == email), :]
        #ind = grade.loc[lambda x: (x['student_id'] == 202013294) & (x['email'] == 'chsm0403@naver.com'), :]
    except Exception as e:
        print(e)
        raise ValueError('해당하는 학번과 이메일이 없습니다.')

    if len(ind) == 0:
        raise ValueError('해당하는 학번과 이메일이 없습니다.')

    return grade, ind

def draw_plot(d: np.array) -> alt.Chart:
    avg, std = np.mean(d), np.std(d, ddof=1)
    d_pdf = pd.DataFrame(
        dict(
            Score=np.linspace(0, 100, 1000),
            PDF=stats.t.pdf(np.linspace(0, 100, 1000), len(d) - 1, avg, std),
        )
    )
    y, x = np.histogram(d, bins=np.linspace(0, 100, 11), density=True)
    d_hist = pd.DataFrame(dict(Score=x[:-1], PDF=y))

    p_pdf = alt.Chart(d_pdf).encode(
        x='Score:Q', y='PDF:Q'
    ).mark_line(
        color='red'
    )
    p_hist = alt.Chart(d_hist).encode(
        x='Score:Q', y='PDF:Q'
    ).mark_bar(
        width=20, opacity=0.7
    )
    p = alt.layer(p_hist + p_pdf).resolve_scale(
        y='independent'
    )
    return p


st.title('성적 미리보기')



