**Table of Contents**:
- [Week 5, 2022](#week-5-2022)
  - [Thursday, February 3, 2022](#thursday-february-3-2022)
    - [Algorithms and Data Structures](#algorithms-and-data-structures)
    - [Python & ML tasks](#python--ml-tasks)
    - [General](#general)
    - [SQL](#sql)
    - [CS224n: Natural Language Processing with Deep Learning](#cs224n-natural-language-processing-with-deep-learning)
      - [Language models](#language-models)
      - [Continuous Bag of Words Model (CBOW)](#continuous-bag-of-words-model-cbow)

# Week 5, 2022
## Thursday, February 3, 2022
### Algorithms and Data Structures
* Converting character to uppercase  
    How to do this?   
    * `ord(char)` takes a character as an argument and outputs the corresponding ASCII code.   
    * `chr(code)` takes an ASCII code and outputs the corresponding character.  
    * Need to subtract `32` from the code of the input character if the character is in range `[97, 122]`.
* Strings matching  
    How to compare strings?
    * Just using `==` operator
* Number of words in the text  
    How to compute the number of words in the text?
    * Just split the words by whitespaces `text.split()` and count the elements in resulting list `len(text.split())`.
* Palindromes  
    How to find out whether the input word is palindrome or not?
    * Reverse string `text[::-1]` and compare it with the original string
### Python & ML tasks
* What will output the next code?  
    ```python
    import numpy as np  
    a = np.linspace(1, 4, 4)  
    b = a.reshape(2, 2)  
    b[0, 0] = 0  
    print(a[0] + b[0, 0] + b[1, 1])
    ```
    The right answer is 4, because `np.reshape()` method returns a new view of original array. So, changing elements in array `b` will lead to changes in array `a`. On the contrary, `np.resize()` will happen in-place and doesn't return anything. 
* Choose the code that will output `[[1,2], [3,4]]`
    ```python
    import numpy as np
    a = np.linspace(1, 4, 4)
    # your code
    print(a)
    ```
    The right answer is `a.resize(2,2)`, because it's changing the shape and the size of the array in-place.
* What will output the next code?
    ```python
    import numpy as np
    x = np.array([[1, 2], [3, 4]])
    a = x.flatten()
    b = x.ravel()
    a[0] = 0
    b[1] = 0
    print(x.sum())
    ```
    It will output 8, because `np.flatten()` returns a copy of array and `np.ravel()` doesn't. 
### General
* What is XOR operation (`^` in Python)?  
    XOR it is a logical operation that is true if and only if its arguments differ (one is true, the other is false).
### SQL
* What is a nested query in SQL? How to write it?  
    A nested query is a query inside another query.  
    A subquery is used to select data that will be used in the condition for selecting records in the main query. It is used for:
    * comparing the expression with the result of the nested query;
    * determining whether an expression is included in the results of a subquery;
    * checking if the query selects certain rows.  

    A subquery has the following components:
    * keyword `SELECT` followed by column names or expressions (most often the list contains one element);
    * the `FROM` keyword and the name of the table from which the data is selected;
    * optional `WHERE` clause;
    * optional `GROUP` `BY` clause:
    * optional `HAVING` clause.  

    Nested queries can be included in `WHERE` or `HAVING` like this (optional elements are indicated in square brackets, one of the elements is separated by |):
    * `WHERE` | `HAVING` expression comparison_operator (subquery);
    * `WHERE` | `HAVING` expression that includes a subquery;
    * `WHERE` | `HAVING` expression [`NOT`] `IN` (subquery);
    * `WHERE` | `HAVING` expression comparison_operator `ANY` | `ALL` (subquery).  

    Subqueries can also be inserted into the main query after the SELECT keyword.

    Example 1:
    ```SQL
    select
        title, author, price, amount
    from 
        book
    where 
        price > (select avg(price) from book) and
        amount < (select avg(amount) from book)
    ```
    Example 2:
    ```SQL
    select
        title,
        author,
        amount,
        (select max(amount) from book) - amount as Заказ
    from 
        book
    where 
        (select max(amount) from book) - amount > 0
    ```
### CS224n: Natural Language Processing with Deep Learning
#### Language models
* What is a language model?  
    It is a model that will assign a probability to a sequence of tokens.  
    Mathematically, we can call this probability on any given sequence of `n` words: $`P(w_1,w_2, ... ,w_n)`$.  
* What is a good language model?  
    "The cat jumped over the puddle."  
    A good language model will give this sentence a high probability because this is a completely valid sentence, syntactically and semantically. 
    Similarly, the sentence "stock boil fish is toy" should have a very low probability because it makes no sense.
* What is the unigram model?  
    We can take the unary language model approach and break apart this probability by assuming the word occurrences are completely independent: $`P(w_1,w_2, ... , w_n) = \prod_{i=1}^{n}{P(w_i)}`$
* What is the bigram model?  
    We let the probability of the sequence depend on the pairwise probability of a word in the sequence and the word next to it. We call this the *bigram model* and represent it as: $`P(w_1,w_2, ... , w_n) = \prod_{i=2}^{n}{P(w_i|w_{i-1})}`$
#### Continuous Bag of Words Model (CBOW)
* What is CBOW model?  
    Predicting a center word from the surrounding context.  
    For each word, we want to learn 2 vectors:
    * v: (input vector) when the word is in the context
    * u: (output vector) when the word is in the center
