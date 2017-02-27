"""
# Model definition.

Defines the database model. The model consists of question-answer
pairs. Each answer is associated with exactly one question.

## Example usage

The example below illustrates how to add new question-answer pairs
to the database.

```python
from model import Session, Question, Answer

session = Session()

answer = Answer(body='body')
question = Question(title='title', body='body', answers=[answer])
session.add(answer)
session.add(question)
session.commit()

session.close()
```
"""

from __future__ import print_function

import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base

from sqlite3 import dbapi2 as sqlite

# Defines model constants.
QUESTION_TITLE_MAXLEN = 100  # characters
QUESTION_BODY_MAXLEN = 1000  # characters
ANSWER_BODY_MAXLEN = 1000  # characters

DB_NAME = 'reddit.db'
ECHO_COMMANDS = False

engine = sql.create_engine('sqlite:///' + DB_NAME,
    module=sqlite,
    echo=ECHO_COMMANDS)

Session = sql.orm.sessionmaker(bind=engine)
Base = declarative_base()


class Question(Base):
  """Defines the Question model."""

  __tablename__ = 'questions'

  id = sql.Column(sql.Integer,
      sql.Sequence('question_id_sequence'),
      primary_key=True)
  title = sql.Column(sql.String(QUESTION_TITLE_MAXLEN))
  body = sql.Column(sql.String(QUESTION_BODY_MAXLEN))

  answers = sql.orm.relationship('Answer',
      backref='question')

  def __repr__(self):
    return self.title


class Answer(Base):
  """Defines the Answer model."""

  __tablename__ = 'answers'

  id = sql.Column(sql.Integer,
      sql.Sequence('answer_id_sequence'),
      primary_key=True)
  body = sql.Column(sql.String(ANSWER_BODY_MAXLEN))
  question_id = sql.Column(sql.Integer,
      sql.ForeignKey('questions.id'))

  def __repr__(self):
    return self.body

# Updates the tables.
Base.metadata.create_all(engine)


if __name__ == '__main__':

  session = Session()

  num_questions = session.query(Question).count()

  # Displays all the questions in the database.
  print('Questions in database:')
  for i, question in enumerate(session.query(Question).limit(10)):
    print('%d / %d: %s' % (i + 1, num_questions, question))
    for answer in question.answers:
      print('  ', answer)

  session.close()

