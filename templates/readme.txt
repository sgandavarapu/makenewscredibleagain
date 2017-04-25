Make News Credible Again (#MNCA)
Sashi Gandavarapu, Brennan Borlaug, Talieh Hajzargarbashi, and Umber Singh

Problem Statement:
We aim to build a model that is capable of discerning whether an article is credible or not based on features derived solely from its text (i.e. word choice, writing style, title, etc.).

Background:
The widespread propagation of false information online (“fake news”) is not a recent phenomenon but its perceived impact in the 2016 U.S. presidential election has thrust the issue into the spotlight. In this project, we explore a number of machine learning-based approaches for solving the problem. Our first step was to identify the various forms of “fake news”. 

Four Common Forms of “Fake News”:
1)	Clickbait — Shocking headlines meant to generate clicks to increase ad revenue. Oftentimes these stories are highly exaggerated or totally false.

2)	Propaganda — Intentionally misleading or deceptive articles meant to promote the author’s agenda. Oftentimes the rhetoric is hateful and incendiary.

3)	Commentary/Opinion — Biased reactions to current events. These articles oftentimes tell the reader how to perceive recent events.

4)	Humor/Satire — Articles written for entertainment. These stories are not meant to be taken seriously.

In this project, we focused on developing a classifier that was able to detect clickbait articles and propaganda articles. 

Data: 

To acquire a sufficiently large labeled corpus of articles to train on, we scraped the websites of both credible and non-credible sources listed in the OpenSources (http://www.opensources.co/) database for new articles daily. Articles were given the same label as their source.

Approach:

1)	Scrape source websites for new article context (text and title) daily and store on cloud server.

2)	Preprocess articles for content-based classification using various widely used techniques in NLP.

3)	Train different machine learning models to classify the news articles 

4)	Create a web application (using Falsk API) to serve as the front-end for our classifier that returns a classification, a confidence metric and few important features in the model.
