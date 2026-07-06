# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

MusicSync

---

## 2. Intended Use  

This recommender is designed to recommend a number of songs based off a given user profile. This user profile will list preferences such as favorite genre,mood, and a target number for the energy, valence, acousticness of the song. The recommender takes this information and then recommends songs that have features similar to the user's preference.

Since this model is fairly simple, it is only intended for classroom exploration.

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Scoring is based off 5 features: genre, energy, mood, acousticness, and valence. If the genre and/or mood of the song is one of the user's favorites, then the score will go up by .3 for genre and .15 for mood. Energy, acousticness, and valence are scored based off of how close the numbers are to each other with closer numbers yielding a higher score. Overall, genre and energy contribute a majority of the scoring weight. 

The scores are calculated for each song in the dataset compared against the user's preferences, and then ranked from highest score to lowest.

---

## 4. Data  

The model uses hardcoded/written data of songs that lists features needed for scoring such as its genre, mood, energy number, valence number, acoustic number, etc. There is a limited number of songs as they are made up.

Due to the limited amount of songs, currently there are only 20 in the dataset, there are many mixes of genres, moods, energy, etc. that could fit a user's preference better. 

---

## 5. Strengths  

The system works well when a user's preference is mostly in line with the songs in the dataset.

---

## 6. Limitations and Bias 

The amount of data available is a big limiter because certain profiles will just not have any songs that suits them so the songs recommended are not that close to what they would like.

There is no way to gauge the similarities between genres and moods so if the song genre/mood is not in the user's favorites, the chances it is recommended decreases greatly.

There is no personalization or feedback loop that allows the recommender to learn from what the user actually likes to listen to.
---

## 7. Evaluation  

I tested three user profiles:
- chill lofi user
- sad metal/rock listener
- high-enery upbeat listener

I looked for songs whose song titles made sense with the genres and scores that were relatively close to .80 or .90

I was not really surprised with the results as there was a very limited amount of data so some users got songs that would not suit them well.

I believe that the scoring system has flaws trying to match genre and mood because those are all or nothing while the other features at least score based on how close the numbers are together. There is simply no way to tell if two or more genres/mood are closely related in this recommender.

---

## 8. Future Work  

Adding more data/songs is needed to improve this recommender. This can maybe be added through an API so I can get real songs. Hardcoding them is unfeasable.

I would also want to add some sort of agentic loop where the system can learn from new songs and user preferences and add that to the system such as new genres and moods so that the recommender is not so limited.

---

## 9. Personal Reflection  

Recommender systems rely a lot on self looping and learning from data. To me, it seems really difficult to be able to create such a system and it is amazing that people were able to figure out how to do it. Even coding something so simple was not easy work and other recommendation systems can do so much more.

