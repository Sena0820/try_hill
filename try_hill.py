from abc import ABC

import numpy as np

import requests
from bs4 import BeautifulSoup
from gensim.models import word2vec
from simpleai.search import SearchProblem, hill_climbing
import random


# この関数はあるクエリで検索した際に指定したサイトで何位上位かを返してくれる
def return_rank(query):
    # 上位から何件までのサイトを抽出するか指定する
    global site_title
    search = query
    target = '初心者がPythonで作れるもの5選！すぐに作れるものを徹底解説'
    how_page = 50 + 1

    print(f'【検索ワード】{search}')

    # Googleから検索結果ページを取得する
    url = f'https://www.google.co.jp/search?hl=ja&num={how_page}&q={search}'
    request = requests.get(url)

    # Googleのページ解析を行う
    soup = BeautifulSoup(request.text, "html.parser")
    search_site_list = soup.select('div.kCrYT > a')
    # num2 = random.randint(1,10)
    # ページ解析と結果の出力
    num = random.randint(1, 10)
    for rank, site in zip(range(1, how_page), search_site_list):

        # site_title = site.select('h3.zBAuLc')[0].text
        try:
            site_title = site.select('h3.zBAuLc')[0].text
        except IndexError:
            # site_title = site.select('img')[0]['alt']
            site_url = site['href'].replace('/url?q=', '')
        # 結果を出力する
        if site_title == target:
            # print('「初心者がPythonで作れるもの5選！すぐに作れるものを徹底解説」' + 'の順位は' + str(rank))
            # print(str(rank) + "位: " + site_title)
            return 50/rank
    return 50/ (50+ num)


model = word2vec.Word2Vec.load("word2vec.gensim.model")

# stateには単語ベクトルを入れる
# まずこのｆｘの代わりにクエリを入力すると順位を返してくれるプログラムを書く

start = model.wv['Perl']


class HillProblem(SearchProblem, ABC):
    def actions(self, state):
        list1 = [state + 0.15, state - 0.15]  # このリストの形を崩して良いのか？
        # newstate = np.average(state + 0.02)  # もう少し考える必要がありそう、平均ベクトルを使うならどうするべきか考える必要がありそう
        # newstate2 = np.average(state - 0.02)
        # list1.append(newstate)  # よくわからないが右に1個進めることを書いている、おそらく最大である20を超えないためのコード
        # list1.append(newstate2)

        print("{} -> [{:.0f} {:.0f}]".format(query, v*50, v*50))
        return list1

    def result(self, state, action):
        return action

    def value(self, state):
        global query
        query = model.wv.similar_by_vector(vector=state, topn=2, restrict_vocab=None)[1][0]
        global v
        v = return_rank(query)
        return v


problem = HillProblem(initial_state=start)  # ここでproblemにクラスを対応させている
result = hill_climbing(problem)  # hill_climbingというメソッドを使っている。引数にproblemを入れている。
# print(return_rank('Python'))
# print(return_rank('Ruby'))
# print(return_rank('Javascript'))
