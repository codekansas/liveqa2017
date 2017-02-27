"""
# Scraper Implementation

This scraper uses the Reddit Python API PRAW to scrape question-answer
pairs from a number of subreddits.
"""

from __future__ import print_function

import praw
import threading

from model import (Session, Question, Answer,
                   QUESTION_TITLE_MAXLEN, QUESTION_BODY_MAXLEN,
                   ANSWER_BODY_MAXLEN)


def _validate_question(q_title, q_body):
  """Determines is a question should be added to the database."""

  return (q_title.endswith('?')
          and len(q_title) < QUESTION_TITLE_MAXLEN
          and len(q_body) < QUESTION_BODY_MAXLEN)


def _validate_answer(a_body):
  """Determines if an answer should be added to the database."""

  return (len(a_body) < ANSWER_BODY_MAXLEN
          and not a_body.startswith('[deleted]'))

def get_qa_pairs(reddit, session, num_questions, num_answers, subreddit):
  """Adds question-answer pairs from the subreddit.
  
  Args:
    reddit: praw.Reddit instance, for communicating with the Reddit API.
    session: Session instance, for writing to the database.
    num_questions: int, number of questions to parse.
    num_answers: int, number of answers to parse.
    subreddit: str, the subreddit to parse from.
  """

  num_parsed = 0
  print('started "%s" thread' % subreddit)

  def _add_answers(question, submission):
    """Adds answers associated with the question."""

    num_parsed = 0

    for comment in submission.comments:
      body = comment.body.strip()
      
      if not _validate_answer(body):
        continue

      answer = Answer(body=body)
      session.add(answer)

      question.answers.append(answer)

      num_parsed += 1
      if num_parsed >= num_answers:
        break

    session.add(question)
    session.commit()

  for submission in reddit.subreddit(subreddit).top('all'):
    title = submission.title.strip()
    body = submission.selftext.strip()

    if not _validate_question(title, body):
      continue

    question = Question(title=title, body=body, answers=[])
    _add_answers(question, submission)

    num_parsed += 1
    if num_parsed % 10 == 0:
      print('parsed %d from "%s"' % (num_parsed, subreddit))
    if num_parsed >= num_questions:
      break


if __name__ == '__main__':
  reddit = praw.Reddit()
  session = Session()
  subreddits = {
      'AskReddit': 1000,
      'sex': 100,
      'CSCareerQuestions': 100,
      'AskWomen': 100,
      'AskMen': 100,
      'relationship_advice': 100,
      'AskHistorians': 100,
  }
  num_answers = 2

  for subreddit, num_questions in subreddits.items():
    thread = threading.Thread(target=get_qa_pairs,
                              args=(reddit, session, num_questions,
                                    num_answers, subreddit))
    thread.daemon = True
    thread.start()

  # Holds until keyboard interrupt.
  import time

  while True:
    time.sleep(10)

