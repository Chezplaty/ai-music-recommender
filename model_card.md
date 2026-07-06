# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  

MusicSync

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

The amount of data available is a big limiter because certain profiles will just not have any songs that suits them so the songs recommended are not that close to what they would like.

There is no way to gauge the similarities between genres and moods so if the song genre/mood is not in the user's favorites, the chances it is recommended decreases greatly.

There is no personalization or feedback loop that allows the recommender to learn from what the user actually likes to listen to.
---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

I tested three user profiles:
- chill lofi user
- sad metal/rock listener
- high-enery upbeat listener

I looked for songs whose song titles made sense with the genres and scores that were relatively close to .80 or .90

I was not really surprised with the results as there was a very limited amount of data so some users got songs that would not suit them well.

I believe that the scoring system has flaws trying to match genre and mood because those are all or nothing while the other features at least score based on how close the numbers are together. There is simply no way to tell if two or more genres/mood are closely related in this recommender.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  
