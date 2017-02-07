# Project Proposal

## Introduction

 - Factoid question answering research has been focused on for many years, but the no-factoid research is still on the go.
   Our team will focus on the no-factoid questions to retrieve the best answer from the existing resources. The goal of doing
   this project is that to participate the TREC LiveQA 2017 competition.

## Outline

 - Build a full question-answering system
   - API for communicating: Takes in question, returns answer
    - Searching data source to check if the questions are duplicated ones
        - Yahoo! Answers
        - Bing Web Search API
        - Twitter API
        - Answers.com
        - WikiHow
   - Build local corpus:
    - Labelled data from TREC LiveQA 2015 and 2016
    - Archives of Yahoo! Answers (http://webscope.sandbox.yahoo.com/)
   - Extract candidates and their context
    - Retrieved relevant answers
   - Communicate with another API: [Yahoo Answers](https://developer.yahoo.com/answers/V1/questionSearch.html), [DuckDuckGo](https://duckduckgo.com/api), [KNGIN](http://www.kngine.com/QAAPI.html)
   - Shallow filtering: Use 1,2,3-gram cosine similarity, TF-IDF or BM-25 to get top-N results
   - Deep filtering: Use semantic-aware method to re-rank top-N results
   - Trained LambdaMART model to optimize NDCG on data from TREC

## Objectives

 - Good performance on past TrecQA datasets
 - Come up with novel ideas for solving this problem
 - Make a submission for the competition
