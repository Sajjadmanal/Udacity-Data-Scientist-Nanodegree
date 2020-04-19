# Data Scientist Nanodegree Udacity
## Recommendations With IBM project

### Introduction
<p>In this project, we will analyze the interactions that users have with articles on the IBM Watson Studio platform, 
 and make recommendations to them about new articles they'll like.</p>
 
### Motivation
For this project I will be looking at the interactions that users have with articles on the IBM Watson Studio platform.

In order to determine which articles to show to each user, It will be performing a study of the data available on the [IBM Watson Studio platform](https://dataplatform.cloud.ibm.com/).
### Libraries
<ul>
  <li>python 3.7 and above</li>
  <li>pandas</li>
  <li>numpy</li>
  <li>matplotlib</li>
  <li>pickle</li>
  <li>re</li>
  <li>nltk</li>
  <li>sklearn</li>
  <li>jupyter</li>
</ul>

### Data
<p>2 .csv files</p>
<ul>
  <li>user-item-interactions.csv: Interactions between users and articles.</li>
  <li>articles_community.csv: Contents of articles.</li>
</ul>

### Overview
#### I. Exploratory Data Analysis
<p>Before making recommendations of any kind, you will need to explore the data you are working with for the project. 
Dive in to see what you can find. There are some basic, required questions to be answered about the data you are 
working with throughout the rest of the notebook. Use this space to explore, before you dive into the details of 
your recommendation system in the later sections.</p>

#### II. Rank Based Recommendations
<p>To get started in building recommendations, you will first find the most popular articles simply based on 
the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles 
with the most interactions are the most popular. These are then the articles we might recommend to new users 
(or anyone depending on what we know about them).</p>

#### III. User-User Based Collaborative Filtering
<p>In order to build better recommendations for the users of IBM's platform, we could look at users that are 
similar in terms of the items they have interacted with. These items could then be recommended to the similar users. 
This would be a step in the right direction towards more personal recommendations for the users</p>

#### IV. Content Based Recommendations
<p>Given the amount of content available for each article, there are a number of different ways in which someone might 
choose to implement a content based recommendations system. Using your NLP skills, you might come up with some extremely 
creative ways to develop a content based recommendation system. You are encouraged to complete a content based 
recommendation system, but not required to do so to complete this project.</p>

#### V. Matrix Factorization
<p>Finally, you will complete a machine learning approach to building recommendations. Using the user-item interactions, 
you will build out a matrix decomposition. Using your decomposition, you will get an idea of how well you can predict new 
articles an individual might interact with (spoiler alert - it isn't great). You will finally discuss which methods you 
might use moving forward, and how you might test how well your recommendations are working for engaging users.</p>

### Reference
<p>Dataset is provided by <a href="https://www.udacity.com/">Udacity</a>. 
It is the part of <a href="https://www.udacity.com/course/data-scientist-nanodegree--nd025">Data Scientist Nanodegree</a>.</p>
